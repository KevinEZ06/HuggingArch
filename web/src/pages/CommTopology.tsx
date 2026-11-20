// The topology sketch of one collective: the GPUs of each node its ranks sit
// on, the ranks the backend published for the group, and — one strip per
// stage — how the chosen algorithm moves the data. Schematic on purpose: a
// wide cluster is elided to a count rather than drawn to scale, and a stage's
// arrows show the shape of the exchange, not a rank-by-rank schedule.
//
// The stages, their order and their words are the backend's
// (`comm_choice.candidates[].stages`). This file only decides which picture a
// stage KIND gets; a kind it has no picture for is drawn as a labelled box,
// so an algorithm the backend adds still draws.
import {
  algorithmColor,
  topologyLayout,
  type CommChoice,
  type CommStage,
  type NodeCell,
} from "../comm";

const BOX_W = 34;
const BOX_H = 26;
const STRIDE = 44;
const NODE_PAD = 10;
const NODE_GAP = 76;
const MARGIN = 8;
/** One stage's strip: its label, then arcs or the switch bar, then the GPU
 *  row, and under it the node names and the ring's arcs. The bands do not
 *  overlap, so a stage never draws over its own labels. */
const STRIP_H = 104;
const ROW_Y = 46;
const SWITCH_Y = 24;
const SWITCH_H = 12;
/** How far an arc above the row may rise — it must clear the strip's label. */
const MAX_LIFT = 18;
const LIMITS = { nodes: 2, perNode: 8 };

function nodeWidth(perRow: number): number {
  return perRow * STRIDE - (STRIDE - BOX_W) + 2 * NODE_PAD;
}

export function CommTopology({ choice, stages }: { choice: CommChoice; stages: CommStage[] }) {
  const layout = topologyLayout(choice.fabric, choice.ranks, LIMITS);
  const drawn: CommStage[] = stages.length
    ? stages
    : [{ kind: "", label: "this algorithm declares no stages" }];
  const perRow = layout.nodes[0]?.ranks.length || 1;
  const nodeW = nodeWidth(perRow);
  const width = MARGIN * 2 + layout.nodes.length * nodeW + (layout.nodes.length - 1) * NODE_GAP;
  const height = drawn.length * STRIP_H + 8;
  const colour = algorithmColor(choice.algorithm);
  const nodeX = (n: number) => MARGIN + n * (nodeW + NODE_GAP);
  const gpuX = (n: number, g: number) => nodeX(n) + NODE_PAD + g * STRIDE;
  return (
    <div data-comm-topology="">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ maxWidth: width, height: "auto" }}
        role="img"
        aria-label={`${choice.op} over ${choice.world} ranks on ${choice.fabric.link}, ${choice.algorithm}`}
      >
        <defs>
          <marker id="comm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0 L8 4 L0 8 z" fill={colour} />
          </marker>
        </defs>
        {drawn.map((stage, i) => (
          <g key={`${stage.kind}-${i}`} transform={`translate(0 ${i * STRIP_H})`} data-comm-stage={stage.kind}>
            <text x={MARGIN} y={12} className="fill-gray-500" style={{ fontSize: 10 }}>
              {drawn.length > 1 ? `${i + 1} · ` : ""}{stage.label}
            </text>
            {layout.nodes.map((node, slot) => (
              <NodeFrame
                key={node.index}
                node={node}
                x={nodeX(slot)}
                w={nodeW}
                gpuX={(g) => gpuX(slot, g)}
                label={i === 0 && layout.nodes.length > 1 ? `node ${node.index}` : null}
              />
            ))}
            <StageArrows
              stage={stage}
              nodes={layout.nodes}
              gpuX={gpuX}
              nodeX={nodeX}
              nodeW={nodeW}
              colour={colour}
            />
          </g>
        ))}
      </svg>
      {(layout.hiddenNodes > 0 || (layout.nodes[0]?.hiddenRanks || 0) > 0) && (
        <div className="text-10 text-gray-400">
          {layout.hiddenNodes > 0 ? `+ ${layout.hiddenNodes} more node${layout.hiddenNodes === 1 ? "" : "s"} like these. ` : ""}
          {(layout.nodes[0]?.hiddenRanks || 0) > 0
            ? `Each node holds ${layout.perNode} GPUs; ${perRow} of them are drawn.`
            : ""}
        </div>
      )}
    </div>
  );
}

/** One node's box and the GPUs in it; the collective's ranks are filled. The
 *  node's name sits UNDER the box: above it is where the stages draw. */
function NodeFrame({ node, x, w, gpuX, label }: {
  node: NodeCell;
  x: number;
  w: number;
  gpuX: (g: number) => number;
  label: string | null;
}) {
  return (
    <g>
      <rect
        x={x}
        y={ROW_Y - NODE_PAD}
        width={w}
        height={BOX_H + 2 * NODE_PAD}
        rx={6}
        fill="#f9fafb"
        stroke="#e5e7eb"
      />
      {label ? (
        <text x={x + 2} y={ROW_Y + BOX_H + NODE_PAD + 11} className="fill-gray-400" style={{ fontSize: 9 }}>
          {label}
        </text>
      ) : null}
      {node.ranks.map((cell, g) => (
        <g key={cell.rank}>
          <rect
            x={gpuX(g)}
            y={ROW_Y}
            width={BOX_W}
            height={BOX_H}
            rx={4}
            fill={cell.inWorld ? "#e0f2fe" : "#ffffff"}
            stroke={cell.inWorld ? "#0284c7" : "#d1d5db"}
            data-rank={cell.rank}
            data-in-world={cell.inWorld ? "1" : undefined}
          />
          <text
            x={gpuX(g) + BOX_W / 2}
            y={ROW_Y + BOX_H / 2 + 3}
            textAnchor="middle"
            className={cell.inWorld ? "fill-sky-800" : "fill-gray-400"}
            style={{ fontSize: 9 }}
          >
            {cell.rank}
          </text>
        </g>
      ))}
    </g>
  );
}

type ArrowProps = {
  stage: CommStage;
  nodes: NodeCell[];
  gpuX: (n: number, g: number) => number;
  nodeX: (n: number) => number;
  nodeW: number;
  colour: string;
};

/** The stage's movement. Each KIND the backend names gets one picture, drawn
 *  over the ranks the group uses — not over every GPU on the box. A kind with
 *  no picture here falls through to a labelled box. */
function StageArrows({ stage, nodes, gpuX, nodeX, nodeW, colour }: ArrowProps) {
  const members: { x: number; node: number }[] = [];
  const perNode = new Map<number, number[]>();
  nodes.forEach((node, slot) => {
    node.ranks.forEach((cell, g) => {
      if (!cell.inWorld) return;
      const x = gpuX(slot, g) + BOX_W / 2;
      members.push({ x, node: slot });
      perNode.set(slot, [...(perNode.get(slot) ?? []), x]);
    });
  });
  const line = (d: string, extra?: { dash?: string; opacity?: number; width?: number }) => (
    <path
      d={d}
      fill="none"
      stroke={colour}
      strokeWidth={extra?.width ?? 1.2}
      strokeDasharray={extra?.dash}
      opacity={extra?.opacity ?? 0.85}
      markerEnd="url(#comm-arrow)"
    />
  );
  const top = ROW_Y - 2;
  const bottom = ROW_Y + BOX_H + 2;

  if (stage.kind === "switch_reduce" || stage.kind === "switch_multicast") {
    const down = stage.kind === "switch_multicast";
    return (
      <g>
        {nodes.map((node, slot) => (
          <g key={node.index}>
            <rect x={nodeX(slot)} y={SWITCH_Y} width={nodeW} height={SWITCH_H} rx={3} fill="#ecfdf5" stroke={colour} />
            <text x={nodeX(slot) + nodeW / 2} y={SWITCH_Y + SWITCH_H - 3} textAnchor="middle" style={{ fontSize: 8 }} className="fill-emerald-700">
              NVSwitch
            </text>
          </g>
        ))}
        {members.map((m, i) => (
          <g key={i}>
            {down
              ? line(`M ${m.x} ${SWITCH_Y + SWITCH_H} L ${m.x} ${top}`, { dash: "3 2" })
              : line(`M ${m.x} ${top} L ${m.x} ${SWITCH_Y + SWITCH_H}`)}
          </g>
        ))}
      </g>
    );
  }

  if (stage.kind === "ring") {
    const arcs = members.slice(0, -1).map((m, i) => {
      const next = members[i + 1];
      const mid = (m.x + next.x) / 2;
      return <g key={i}>{line(`M ${m.x} ${bottom} Q ${mid} ${bottom + 14} ${next.x - 4} ${bottom}`)}</g>;
    });
    const first = members[0];
    const last = members[members.length - 1];
    const wrap = first && last && first !== last
      ? line(`M ${last.x} ${bottom} Q ${(first.x + last.x) / 2} ${bottom + 30} ${first.x + 4} ${bottom}`, { dash: "4 3", opacity: 0.6 })
      : null;
    return <g>{arcs}{wrap}</g>;
  }

  if (stage.kind === "all_to_all") {
    // Every rank writes to every peer. Above ~8 ranks the full fan is a
    // smear, so one rank's writes stand for all of them (said below).
    const dense = members.length > 8;
    const senders = dense ? members.slice(0, 1) : members;
    return (
      <g>
        {senders.flatMap((from, i) =>
          members
            .filter((to) => to !== from)
            .map((to, j) => {
              const mid = (from.x + to.x) / 2;
              const lift = Math.min(MAX_LIFT, 6 + Math.abs(to.x - from.x) / 12);
              return (
                <g key={`${i}-${j}`}>
                  {line(`M ${from.x} ${top} Q ${mid} ${top - lift} ${to.x} ${top}`, { opacity: dense ? 0.6 : 0.3, width: 1 })}
                </g>
              );
            }),
        )}
        {dense ? (
          <text x={MARGIN} y={23} className="fill-gray-400" style={{ fontSize: 9 }}>
            one rank&apos;s writes shown; every rank does the same
          </text>
        ) : null}
      </g>
    );
  }

  if (stage.kind === "reduce_scatter" || stage.kind === "all_gather") {
    // One slice's owner stands for all of them: in reduce-scatter the peers'
    // contributions land on it, in all-gather its reduced slice goes back
    // out. A hierarchical schedule does this INSIDE each node, so each node
    // gets its own owner and no arc crosses the gap between the boxes.
    const inward = stage.kind === "reduce_scatter";
    return (
      <g>
        {[...perNode.values()].flatMap((xs, n) => {
          const hub = xs[0];
          if (hub === undefined) return [];
          return xs.slice(1).map((x, i) => {
            const mid = (hub + x) / 2;
            const lift = Math.min(MAX_LIFT, 8 + Math.abs(x - hub) / 10);
            return (
              <g key={`${n}-${i}`}>
                {inward
                  ? line(`M ${x} ${top} Q ${mid} ${top - lift} ${hub} ${top}`, { opacity: 0.6 })
                  : line(`M ${hub} ${top} Q ${mid} ${top - lift} ${x} ${top}`, { opacity: 0.6 })}
              </g>
            );
          });
        })}
      </g>
    );
  }

  if (stage.kind === "inter_node" && nodes.length > 1) {
    const x1 = nodeX(0) + nodeW;
    const x2 = nodeX(1);
    const y = ROW_Y + BOX_H / 2;
    return (
      <g>
        {line(`M ${x1 + 4} ${y - 5} L ${x2 - 4} ${y - 5}`)}
        {line(`M ${x2 - 4} ${y + 5} L ${x1 + 4} ${y + 5}`)}
      </g>
    );
  }

  // A kind this build has no picture for (including an inter-node hop with
  // only one node box drawn): say what it is over the ranks it runs on.
  const left = members[0]?.x ?? nodeX(0);
  const right = members[members.length - 1]?.x ?? nodeX(0) + nodeW;
  return (
    <g>
      <rect
        x={left - BOX_W / 2 - 4}
        y={SWITCH_Y}
        width={Math.max(60, right - left + BOX_W + 8)}
        height={SWITCH_H + 4}
        rx={3}
        fill="none"
        stroke={colour}
        strokeDasharray="4 3"
        opacity={0.7}
      />
      <text
        x={left - BOX_W / 2}
        y={SWITCH_Y + SWITCH_H}
        style={{ fontSize: 8 }}
        className="fill-gray-500"
      >
        {stage.kind || "stage"}
      </text>
    </g>
  );
}
