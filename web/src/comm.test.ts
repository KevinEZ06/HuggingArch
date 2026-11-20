// The reading rules of a collective's published choice. The ones a refactor
// can quietly reverse come first:
//
//  - a row already covers `count` instances and its time already includes all
//    of them, so a group's launches are the SUM of the rows' counts and its
//    step total the sum of the rows' times — never launches × one row's time;
//  - what the card shows per launch is a row-wide figure divided ONCE, by
//    that same count: `t_s`, `wire_bytes_per_rank` and the curves are
//    row-wide while `payload_bytes` and `hops` are already per instance;
//  - the envelope is READ, never taken here. Which algorithm wins where is
//    the pricing rule's own answer; a second one computed in the browser
//    would be a second answer.
//
// The fixtures below are real `comm_choice` payloads (Kimi-K3 on 8×B300 and
// on 16×B300 across two nodes, sglang, 4096-token prefill), with the curves
// thinned to five of the backend's grid points.
import { describe, expect, test } from "vitest";

import {
  chosenCandidate,
  commGroups,
  envelopeCurve,
  feasibleCandidates,
  hasCommChoice,
  perLaunch,
  phaseCommRows,
  topologyLayout,
  type CommChoice,
} from "./comm";
import type { Estimate, ModuleFoldRow } from "./pages/ResultSummary";

export const AR8: CommChoice = {
  op: "all_reduce",
  world: 8,
  count: 3,
  ranks: [0, 1, 2, 3, 4, 5, 6, 7],
  payload_bytes: 58720256.0,
  fabric: {
    link: "NVLink",
    nvls: true,
    gpus_per_node: 8,
    nodes_spanned: 1
  },
  algorithm: "nvls",
  candidates: [
    {
      algorithm: "one_shot",
      description: "each rank writes its whole payload to every peer, then sums the copies it received itself",
      stages: [
        {
          kind: "all_to_all",
          label: "every rank writes its payload to every peer and reduces the copies locally"
        }
      ],
      feasible: true,
      reason: null,
      t_s: 0.0013761393066666666,
      wire_bytes_per_rank: 1233125376.0,
      hops: 1,
      curve: [[1024, 6.023893333333333e-06], [1048576, 3.0466773333333334e-05], [18831788, 0.0004454083866666667], [59787201, 0.0014010346899999998], [1073741824, 0.025059975893333333]]
    },
    {
      algorithm: "two_shot",
      description: "each rank sums one 1/N slice gathered straight from every peer, then writes its summed slice back to every peer",
      stages: [
        {
          kind: "reduce_scatter",
          label: "each rank reduces its 1/N slice from every peer"
        },
        {
          kind: "all_gather",
          label: "each rank writes its reduced slice to every peer"
        }
      ],
      feasible: true,
      reason: null,
      t_s: 0.0003545348266666667,
      wire_bytes_per_rank: 308281344.0,
      hops: 2,
      curve: [[1024, 1.2005973333333334e-05], [1048576, 1.8116693333333333e-05], [18831788, 0.00012185209666666668], [59787201, 0.0003607586725], [1073741824, 0.006275493973333333]]
    },
    {
      algorithm: "nvls",
      description: "each rank sends its slice into the NVSwitch, which reduces it and multicasts the sum back to every rank",
      stages: [
        {
          kind: "switch_reduce",
          label: "the switch reduces each 1/N slice across the ranks"
        },
        {
          kind: "switch_multicast",
          label: "the switch multicasts each reduced slice to every rank"
        }
      ],
      feasible: true,
      reason: null,
      t_s: 0.00018326741333333333,
      wire_bytes_per_rank: 154140672.0,
      hops: 2,
      curve: [[1024, 1.2002986666666666e-05], [1048576, 1.5058346666666667e-05], [18831788, 6.692604833333334e-05], [59787201, 0.00018637933624999998], [1073741824, 0.0031437469866666666]]
    },
    {
      algorithm: "nccl",
      description: "NCCL's own schedule: a ring inside one node; across nodes a reduce-scatter inside each node, the shards exchanged between nodes, then an all-gather inside each node",
      stages: [
        {
          kind: "ring",
          label: "all reduce over NVLink, 8 ranks"
        }
      ],
      feasible: true,
      reason: null,
      t_s: 0.0003785348266666667,
      wire_bytes_per_rank: 308281344.0,
      hops: 6,
      curve: [[1024, 3.6005973333333337e-05], [1048576, 4.2116693333333333e-05], [18831788, 0.00014585209666666667], [59787201, 0.0003847586725], [1073741824, 0.006299493973333333]]
    }
  ]
};

export const AR16: CommChoice = {
  op: "all_reduce",
  world: 16,
  count: 1,
  ranks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  payload_bytes: 58720256.0,
  fabric: {
    link: "RDMA",
    nvls: true,
    gpus_per_node: 8,
    nodes_spanned: 2
  },
  algorithm: "nccl",
  candidates: [
    {
      algorithm: "one_shot",
      description: "each rank writes its whole payload to every peer, then sums the copies it received itself",
      stages: [
        {
          kind: "all_to_all",
          label: "every rank writes its payload to every peer and reduces the copies locally"
        }
      ],
      feasible: false,
      reason: "the group spans 2 nodes; only NCCL's schedule runs across nodes",
      t_s: null,
      wire_bytes_per_rank: null,
      hops: null,
      curve: [

      ]
    },
    {
      algorithm: "two_shot",
      description: "each rank sums one 1/N slice gathered straight from every peer, then writes its summed slice back to every peer",
      stages: [
        {
          kind: "reduce_scatter",
          label: "each rank reduces its 1/N slice from every peer"
        },
        {
          kind: "all_gather",
          label: "each rank writes its reduced slice to every peer"
        }
      ],
      feasible: false,
      reason: "the group spans 2 nodes; only NCCL's schedule runs across nodes",
      t_s: null,
      wire_bytes_per_rank: null,
      hops: null,
      curve: [

      ]
    },
    {
      algorithm: "nvls",
      description: "each rank sends its slice into the NVSwitch, which reduces it and multicasts the sum back to every rank",
      stages: [
        {
          kind: "switch_reduce",
          label: "the switch reduces each 1/N slice across the ranks"
        },
        {
          kind: "switch_multicast",
          label: "the switch multicasts each reduced slice to every rank"
        }
      ],
      feasible: false,
      reason: "the group spans 2 nodes; only NCCL's schedule runs across nodes",
      t_s: null,
      wire_bytes_per_rank: null,
      hops: null,
      curve: [

      ]
    },
    {
      algorithm: "nccl",
      description: "NCCL's own schedule: a ring inside one node; across nodes a reduce-scatter inside each node, the shards exchanged between nodes, then an all-gather inside each node",
      stages: [
        {
          kind: "reduce_scatter",
          label: "reduce scatter over NVLink, 8 ranks"
        },
        {
          kind: "inter_node",
          label: "all reduce over RDMA, 2 ranks"
        },
        {
          kind: "all_gather",
          label: "all gather over NVLink, 8 ranks"
        }
      ],
      feasible: true,
      reason: null,
      t_s: 0.00020357859555555558,
      wire_bytes_per_rank: 110100480.0,
      hops: 8,
      curve: [[1024, 1.600327111111111e-05], [1048576, 1.9349617777777777e-05], [18831788, 7.615710055555556e-05], [59787201, 0.00020698689208333335], [1073741824, 0.0034460086044444446]]
    }
  ]
};

function commRow(choice: CommChoice, over: Partial<ModuleFoldRow> = {}): ModuleFoldRow {
  return {
    op: choice.op,
    display_path: "comm4__all_reduce__on__attn_residual",
    is_comm: true,
    comm_world: choice.world,
    comm_bw_label: "NVLink",
    comm_bw_bytes_per_sec: 9e11,
    // The row's own time: the chosen candidate's, over all `count` instances.
    t_comm_s: 1.8326741333333333e-4,
    comm_choice: choice,
    ...over,
  };
}

function estimate(prefill: ModuleFoldRow[], decode: ModuleFoldRow[] = []): Estimate {
  return { phases: { prefill: { flops_by_module: prefill }, decode: { flops_by_module: decode } } };
}

describe("commGroups", () => {
  test("launches are the sum of the rows' own counts, and the step total the sum of their times", () => {
    const groups = commGroups([
      commRow(AR8, { display_path: "layer.0.comm" }),
      commRow(AR8, { display_path: "layer.1.comm" }),
      commRow({ ...AR8, count: 1 }, { display_path: "layer.2.comm", t_comm_s: 6.108913777777778e-5 }),
    ]);
    expect(groups).toHaveLength(1);
    // 3 + 3 + 1 instances — not 3 rows, and not 3 × the first row's count.
    expect(groups[0].launches).toBe(7);
    expect(groups[0].t_total_s).toBeCloseTo(2 * 1.8326741333333333e-4 + 6.108913777777778e-5, 12);
    expect(groups[0].paths).toEqual(["layer.0.comm", "layer.1.comm", "layer.2.comm"]);
  });

  test("a different world, payload or chosen algorithm is a different group, biggest first", () => {
    const groups = commGroups([
      commRow(AR16, { t_comm_s: 2.0357859555555558e-4 }),
      commRow(AR8),
      commRow(AR8),
    ]);
    expect(groups.map((g) => g.choice.algorithm)).toEqual(["nvls", "nccl"]);
    expect(groups[0].launches).toBe(6);
    expect(groups[1].launches).toBe(1);
  });

  test("a row without a published choice is left out rather than reconstructed", () => {
    expect(commGroups([commRow(AR8, { comm_choice: undefined })])).toEqual([]);
    expect(commGroups(undefined)).toEqual([]);
    expect(hasCommChoice(estimate([commRow(AR8, { comm_choice: undefined })]))).toBe(false);
    expect(hasCommChoice(estimate([], [commRow(AR8)]))).toBe(true);
    expect(hasCommChoice({})).toBe(false);
    expect(phaseCommRows(estimate([commRow(AR8)]), "prefill")).toHaveLength(1);
    expect(phaseCommRows({}, "decode")).toEqual([]);
  });

  test("a row that published no time leaves the group's total unpublished, not zero", () => {
    expect(commGroups([commRow(AR8, { t_comm_s: undefined })])[0].t_total_s).toBeNull();
  });
});

describe("perLaunch", () => {
  test("a row-wide figure divided by the instances it was priced over", () => {
    const chosen = chosenCandidate(AR8);
    expect(AR8.count).toBe(3);
    // The row's time IS the chosen candidate's, so one launch is a third.
    expect(chosen?.t_s).toBeCloseTo(1.8326741333333333e-4, 15);
    expect(perLaunch(chosen?.t_s, AR8.count)).toBeCloseTo(6.108913777777778e-5, 15);
    expect(perLaunch(chosen?.wire_bytes_per_rank, AR8.count)).toBeCloseTo(51380224, 6);
    // hops and payload are already per instance: nothing divides them.
    expect(chosen?.hops).toBe(2);
    expect(AR8.payload_bytes).toBe(58720256);
  });

  test("nothing to divide, or nothing to divide by, is not a number", () => {
    expect(perLaunch(null, 3)).toBeNull();
    expect(perLaunch(undefined, 3)).toBeNull();
    expect(perLaunch(1, 0)).toBeNull();
  });
});

describe("candidates", () => {
  test("a refused candidate keeps the backend's reason and contributes no curve", () => {
    const out = AR16.candidates.filter((c) => !c.feasible);
    expect(out.map((c) => c.algorithm)).toEqual(["one_shot", "two_shot", "nvls"]);
    expect(out[0].reason).toContain("spans 2 nodes");
    expect(out.every((c) => c.curve.length === 0 && c.t_s === null && c.hops === null)).toBe(true);
    expect(feasibleCandidates(AR16).map((c) => c.algorithm)).toEqual(["nccl"]);
  });

  test("one-shot is a normal candidate that simply loses at this payload", () => {
    const one = AR8.candidates.find((c) => c.algorithm === "one_shot");
    expect(one?.feasible).toBe(true);
    expect(one?.reason).toBeNull();
    expect(one!.t_s!).toBeGreaterThan(chosenCandidate(AR8)!.t_s!);
  });

  test("the words and the stages are the backend's, not this build's", () => {
    expect(chosenCandidate(AR8)?.description).toContain("NVSwitch");
    expect(chosenCandidate(AR8)?.stages.map((s) => s.kind)).toEqual(["switch_reduce", "switch_multicast"]);
    expect(chosenCandidate(AR16)?.stages.map((s) => s.kind))
      .toEqual(["reduce_scatter", "inter_node", "all_gather"]);
    expect(chosenCandidate(AR16)?.stages[1].label).toContain("RDMA");
  });

  test("an algorithm no candidate describes yields nothing to show, not another candidate", () => {
    expect(chosenCandidate({ ...AR8, algorithm: "future_kernel" })).toBeNull();
  });
});

describe("envelopeCurve", () => {
  test("the envelope is read, and a payload that publishes none draws none", () => {
    expect(envelopeCurve(AR8)).toEqual([]);
    const published: CommChoice = {
      ...AR8,
      envelope_curve: [[1024, 2.0079644444444444e-6, "one_shot"], [59787201, 4.6e-4, "nvls"]],
    };
    expect(envelopeCurve(published)).toHaveLength(2);
    expect(envelopeCurve(published)[1][2]).toBe("nvls");
  });
});

describe("topologyLayout", () => {
  test("the published ranks are what the sketch marks", () => {
    const l = topologyLayout(AR8.fabric, AR8.ranks);
    expect(l.nodes).toHaveLength(1);
    expect(l.nodes[0].ranks.map((r) => r.rank)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(l.nodes[0].ranks.every((r) => r.inWorld)).toBe(true);
  });

  test("a strided group marks its own ranks, not the first `world` GPUs", () => {
    const l = topologyLayout(AR8.fabric, [0, 2, 4, 6]);
    expect(l.nodes[0].ranks.filter((r) => r.inWorld).map((r) => r.rank)).toEqual([0, 2, 4, 6]);
  });

  test("ranks keep their global numbers across nodes, and extra nodes are a count", () => {
    const ranks = Array.from({ length: 32 }, (_, i) => i);
    const l = topologyLayout({ link: "RDMA", nvls: false, gpus_per_node: 8, nodes_spanned: 4 }, ranks);
    expect(l.nodes).toHaveLength(2);
    expect(l.nodes[1].ranks[0].rank).toBe(8);
    expect(l.hiddenNodes).toBe(2);
  });

  test("the drawn window follows the group up a node too wide to draw", () => {
    const l = topologyLayout(
      { link: "NVLink", nvls: true, gpus_per_node: 72, nodes_spanned: 1 },
      [64, 65, 66, 67],
    );
    expect(l.perNode).toBe(72);
    expect(l.nodes[0].hiddenRanks).toBe(64);
    // The window starts at the group's first member rather than at rank 0,
    // so a high-numbered group is drawn where it is.
    expect(l.nodes[0].ranks[0].rank).toBe(64);
    expect(l.nodes[0].ranks.filter((r) => r.inWorld).map((r) => r.rank)).toEqual([64, 65, 66, 67]);
  });

  test("no published ranks: the nodes are drawn, nothing is marked", () => {
    const l = topologyLayout(AR8.fabric, []);
    expect(l.nodes).toHaveLength(1);
    expect(l.nodes[0].ranks.some((r) => r.inWorld)).toBe(false);
  });
});
