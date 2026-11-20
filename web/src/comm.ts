// What a collective costs is a choice. Each one runs on the fastest algorithm
// the card's fabric can actually run, and the backend publishes that choice on
// the row (`comm_choice`): which algorithm it took, which candidates were
// feasible, why the others were not, what each one does in a sentence, its
// stages in order, and its priced curve over payload size.
//
// This module is the reading side of that fact. It groups the rows that
// describe the same collective and converts the row-wide figures to one
// launch. Nothing here prices anything and nothing here knows what an
// algorithm IS: the names, the sentences and the stages are the backend's, so
// an algorithm it adds reads correctly here without a frontend change. Only
// the colours are the frontend's, and only because they are drawing.
import type { ModuleFoldRow, Estimate } from "./pages/ResultSummary";
import type { FormulaFields } from "./tooltip";

/** The fabric the collective runs on, as the backend read it off the card
 *  and the placement — hardware facts only, no algorithm choice in here. */
export type CommFabric = {
  /** The link the ranks talk over: "NVLink", "PCIe", "RDMA", … */
  link: string;
  /** Whether the NVSwitch can reduce in the fabric (NVLS / SHARP). */
  nvls: boolean;
  gpus_per_node: number;
  nodes_spanned: number;
};

/** One stage of an algorithm's data movement, in the order it runs. `kind` is
 *  the backend's closed vocabulary (`comm_volume.STAGE_KINDS`) and is kept
 *  open here on purpose: a kind this build has no picture for is drawn as a
 *  labelled box rather than refused. */
export type CommStage = { kind: string; label: string };

/** One algorithm the backend considered for this collective.
 *
 *  Units, as the backend states them: `hops` is what ONE instance waits on,
 *  while `t_s`, `wire_bytes_per_rank` and the `curve`'s times are over the
 *  row's `count` instances. Infeasible ones carry the reason, no curve and no
 *  figures. */
export type CommCandidate = {
  algorithm: string;
  /** What this algorithm does, in the backend's own sentence. */
  description: string;
  stages: CommStage[];
  feasible: boolean;
  reason: string | null;
  /** A caveat about this candidate's own price, where the backend states one
   *  (NVLS's coefficient reads optimistic). */
  note?: string | null;
  t_s: number | null;
  wire_bytes_per_rank: number | null;
  hops: number | null;
  /** `[payload_bytes per instance, t_s over the row]` on the backend's grid. */
  curve: [number, number][];
};

/** The envelope the backend priced this collective under: time = min over the
 *  algorithms the fabric can run. `algorithm` is the one that won. */
export type CommChoice = {
  op: string;
  world: number;
  /** How many instances of this collective the row covers. */
  count: number;
  /** The group holding rank 0, as the mesh places it. */
  ranks: number[];
  /** Bytes of ONE instance's payload. */
  payload_bytes: number;
  fabric: CommFabric;
  algorithm: string;
  candidates: CommCandidate[];
  /** The envelope itself where the backend publishes it: `[payload_bytes,
   *  t_s over the row, the algorithm that won there]`. Absent → the card
   *  draws the candidates alone and no envelope band; it does not take the
   *  minimum itself, which would be a second pricing path. */
  envelope_curve?: [number, number, string][];
};

/** Every launch of the same collective in one step: same op, same world, same
 *  payload, same chosen algorithm. The group's own numbers are either read off
 *  one row (they agree by construction) or summed over the rows. */
export type CommGroup = {
  key: string;
  choice: CommChoice;
  /** Instances the step runs: Σ of the rows' own `count`. A row already
   *  covers `count` of them and its time already includes all of them, so
   *  this is a sum over rows and never a product with one row's time. */
  launches: number;
  /** Σ of the rows' published `t_comm_s`; null when no row published one. */
  t_total_s: number | null;
  /** The bandwidth the backend priced against, as it labelled it. */
  bw_label: string | null;
  bw_bytes_per_sec: number | null;
  /** The backend's `t_comm` popover for this row. */
  formula: FormulaFields | null;
  /** Where in the step the launches sit, as the rows name themselves. */
  paths: string[];
};

function groupKey(c: CommChoice): string {
  return `${c.op}|${c.world}|${c.payload_bytes}|${c.algorithm}`;
}

/** A row-wide figure as ONE launch's: the backend prices `count` instances
 *  together, so the card divides once, here, and says so where it draws. */
export function perLaunch(value: number | null | undefined, count: number): number | null {
  if (value == null || !Number.isFinite(value) || !(count > 0)) return null;
  return value / count;
}

/** The collectives of one phase's breakdown, one entry per distinct
 *  collective, biggest share of the step first. Rows without a published
 *  choice are left out: the card draws the envelope or nothing, it does not
 *  reconstruct one. */
export function commGroups(rows: ModuleFoldRow[] | null | undefined): CommGroup[] {
  const groups = new Map<string, CommGroup>();
  for (const row of rows || []) {
    const choice = row.comm_choice;
    if (!choice) continue;
    const key = groupKey(choice);
    let group = groups.get(key);
    if (!group) {
      group = {
        key,
        choice,
        launches: 0,
        t_total_s: null,
        bw_label: row.comm_bw_label ?? null,
        bw_bytes_per_sec: row.comm_bw_bytes_per_sec ?? null,
        formula: row.t_comm_formula ?? null,
        paths: [],
      };
      groups.set(key, group);
    }
    group.launches += choice.count > 0 ? choice.count : 1;
    if (row.t_comm_s != null && Number.isFinite(row.t_comm_s)) {
      group.t_total_s = (group.t_total_s ?? 0) + row.t_comm_s;
    }
    const path = row.display_path || row.name;
    if (path) group.paths.push(path);
  }
  return [...groups.values()].sort((a, b) => (b.t_total_s ?? 0) - (a.t_total_s ?? 0));
}

/** The breakdown rows of a phase — where a collective publishes its choice. */
export function phaseCommRows(result: Estimate, phase: "prefill" | "decode"): ModuleFoldRow[] {
  return result.phases?.[phase]?.flops_by_module || [];
}

/** Whether the step has a collective that published its choice. A single-rank
 *  deployment has no collective at all; a payload from before the backend
 *  published `comm_choice` has collectives with nothing to open. Both answer
 *  false, and the card stays off the page. */
export function hasCommChoice(result: Estimate): boolean {
  return (["prefill", "decode"] as const)
    .some((p) => phaseCommRows(result, p).some((r) => !!r.comm_choice));
}

/** The candidate the backend chose; null when the row names an algorithm no
 *  candidate describes (then the card shows no per-launch figures rather than
 *  inventing them). */
export function chosenCandidate(choice: CommChoice): CommCandidate | null {
  return (choice.candidates || []).find((c) => c.algorithm === choice.algorithm) ?? null;
}

export function feasibleCandidates(choice: CommChoice): CommCandidate[] {
  return (choice.candidates || []).filter((c) => c.feasible && (c.curve || []).length > 0);
}

/** The envelope as the backend published it. The card does NOT take the
 *  minimum of the candidates itself: which algorithm wins where is the
 *  pricing rule's own answer, and a second one computed here would be a
 *  second answer. No envelope published → no band drawn. */
export function envelopeCurve(choice: CommChoice): [number, number, string][] {
  return choice.envelope_curve || [];
}

/** One colour per algorithm wherever it is drawn — the envelope chart's lines,
 *  the group list's badge, the candidate list's swatch. A name this build has
 *  not seen takes a slot from the spare palette by position, so two unknown
 *  names stay apart. Drawing only: nothing here decides anything. */
const ALGORITHM_COLOR: Record<string, string> = {
  one_shot: "#7c3aed",
  two_shot: "#0284c7",
  nvls: "#059669",
  nccl: "#d97706",
};

const SPARE_COLORS = ["#be185d", "#4d7c0f", "#b45309", "#1d4ed8"];

export function algorithmColor(algorithm: string, index = 0): string {
  return ALGORITHM_COLOR[algorithm] ?? SPARE_COLORS[index % SPARE_COLORS.length];
}

/** The op as a reader says it: `all_reduce` → "all-reduce". */
export function collectiveLabel(op: string): string {
  return op.replace(/_/g, "-");
}

/** The fabric in one line: the link, whether the switch can reduce, and how
 *  wide the domain is. */
export function fabricLabel(fabric: CommFabric): string {
  const parts = [fabric.link];
  if (fabric.nvls) parts.push("NVSwitch reduces in-fabric (NVLS)");
  if (fabric.gpus_per_node > 0) parts.push(`${fabric.gpus_per_node} GPUs / node`);
  parts.push(fabric.nodes_spanned > 1 ? `${fabric.nodes_spanned} nodes spanned` : "one node");
  return parts.join(" · ");
}

// --- Where the ranks sit -------------------------------------------------

export type RankCell = { rank: number; inWorld: boolean };
export type NodeCell = { index: number; ranks: RankCell[]; hiddenRanks: number };

/** The boxes the sketch draws: one box per node the collective's ranks sit
 *  on, the GPUs inside it, and which of them are this group's ranks. A wide
 *  cluster is elided rather than drawn to scale — the sketch is schematic, and
 *  128 boxes say less than 16 boxes and a count. */
export type TopologyLayout = {
  nodes: NodeCell[];
  hiddenNodes: number;
  /** GPUs per node as the fabric states it (not the number drawn). */
  perNode: number;
};

/** `ranks` is the group the backend published (the one holding rank 0, its
 *  members `base_stride` apart) — the sketch marks those, never "the first
 *  `world` GPUs", which is a different set under any stride. */
export function topologyLayout(
  fabric: CommFabric,
  ranks: number[],
  limits: { nodes: number; perNode: number } = { nodes: 2, perNode: 8 },
): TopologyLayout {
  const members = new Set(ranks || []);
  const perNode = fabric.gpus_per_node > 0
    ? Math.floor(fabric.gpus_per_node)
    : Math.max(1, members.size);
  const shownPerNode = Math.min(perNode, Math.max(1, limits.perNode));
  const touched = members.size
    ? [...new Set([...members].map((r) => Math.floor(r / perNode)))].sort((a, b) => a - b)
    : Array.from({ length: Math.max(1, Math.floor(fabric.nodes_spanned || 1)) }, (_, i) => i);
  const shown = touched.slice(0, Math.max(1, limits.nodes));
  const nodes: NodeCell[] = shown.map((index) => {
    // The window starts at the node's first member, so a group living high up
    // a 72-GPU node is drawn where it is rather than off the left edge.
    const within = [...members].filter((r) => Math.floor(r / perNode) === index)
      .map((r) => r - index * perNode);
    const first = within.length ? Math.min(...within) : 0;
    const start = Math.max(0, Math.min(first, perNode - shownPerNode));
    return {
      index,
      ranks: Array.from({ length: shownPerNode }, (_, g) => {
        const rank = index * perNode + start + g;
        return { rank, inWorld: members.has(rank) };
      }),
      hiddenRanks: perNode - shownPerNode,
    };
  });
  return { nodes, hiddenNodes: touched.length - shown.length, perNode };
}
