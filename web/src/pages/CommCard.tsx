// Communication: what each collective of the step actually runs, and why.
//
// A collective's time is an envelope — the fastest of the algorithms the
// card's fabric can run, with feasibility decided from hardware facts. Until
// now that choice was invisible: the per-op table showed a `t_comm` and no
// reason for it. This card opens it: the topology the ranks sit on, the
// chosen algorithm's data movement stage by stage, every candidate's priced
// curve, and the numbers of one launch.
//
// Everything is read off `comm_choice` on the breakdown row — including the
// algorithms' names, their one-sentence descriptions and their stages. The
// card picks no algorithm, prices nothing, and takes no minimum of its own.
// When the payload carries no `comm_choice` (TP=1, or a backend that does not
// publish it yet) it draws nothing.
//
// Units: the backend prices a row's `count` instances together, so `t_s`,
// `wire_bytes_per_rank` and the curves are row-wide while `payload_bytes` and
// `hops` are per instance. Everything below is shown PER LAUNCH — divided
// once, in `perLaunch` — and the chart's y axis says so.
import { Fragment, useState, type ReactNode } from "react";
import type { EChartsOption } from "echarts";
import { AXIS, HaChart, interactiveBits } from "../charts/HaChart";
import { FormulaHover } from "../FormulaHover";
import { formatBytes, fmtDuration, fmtTime, formatTokens } from "../format";
import {
  algorithmColor,
  chosenCandidate,
  collectiveLabel,
  commGroups,
  envelopeCurve,
  fabricLabel,
  feasibleCandidates,
  perLaunch,
  phaseCommRows,
  type CommCandidate,
  type CommGroup,
} from "../comm";
import { CommTopology } from "./CommTopology";
import type { Estimate } from "./ResultSummary";

type Phase = "prefill" | "decode";

/** The one thing a reader must know before reading any number here. */
const PRICING_NOTE =
  "Collectives are priced at the fabric's best algorithm, the same for every framework; "
  + "a framework that falls back to a slower path is not modelled.";

export default function CommCard({ result }: { result: Estimate }) {
  const byPhase: Record<Phase, CommGroup[]> = {
    prefill: commGroups(phaseCommRows(result, "prefill")),
    decode: commGroups(phaseCommRows(result, "decode")),
  };
  const phases = (["prefill", "decode"] as const).filter((p) => byPhase[p].length > 0);
  const [phase, setPhase] = useState<Phase>("decode");
  const [pickedKey, setPicked] = useState<string | null>(null);
  // No collective in either phase (a single-rank deployment, or a payload
  // from before the backend published the choice): no card.
  if (!phases.length) return null;
  const tab: Phase = phases.includes(phase) ? phase : phases[0];
  const groups = byPhase[tab];
  const group = groups.find((g) => g.key === pickedKey) ?? groups[0];
  const tabCls = (k: Phase) =>
    "px-3 py-1.5 text-xs font-medium " + (tab === k ? "bg-amber-500 text-white" : "bg-white text-gray-600 hover:bg-gray-50");
  return (
    <section className="card p-5 mb-6" id="infer-comm">
      <div className="flex items-baseline justify-between mb-1 flex-wrap gap-2">
        <h3 className="font-semibold text-gray-900">
          Communication{" "}
          <span className="text-xs text-gray-400 font-normal">
            each collective at the fastest algorithm its fabric can run
          </span>
        </h3>
        <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden" role="tablist" aria-label="Communication phase">
          {phases.map((p) => (
            <button
              key={p}
              id={`infer-comm-tab-${p}`}
              type="button"
              role="tab"
              aria-selected={tab === p}
              className={tabCls(p)}
              onClick={() => { setPhase(p); setPicked(null); }}
            >
              {p === "prefill" ? "Prefill" : "Decode"}
            </button>
          ))}
        </div>
      </div>
      <div className="text-11 text-amber-700 mb-3" id="infer-comm-pricing-note">⚠ {PRICING_NOTE}</div>
      <div className="flex flex-wrap gap-1.5 mb-3" data-comm-groups="">
        {groups.map((g) => {
          const on = g.key === group.key;
          return (
            <button
              key={g.key}
              type="button"
              data-comm-group={g.key}
              aria-pressed={on}
              onClick={() => setPicked(g.key)}
              className={`rounded-lg border px-2.5 py-1 text-left text-11 ${
                on ? "border-amber-400 bg-amber-50" : "border-gray-200 bg-white hover:bg-gray-50"
              }`}
            >
              <span className="font-mono text-gray-800">{collectiveLabel(g.choice.op)}</span>
              <span className="ml-1.5 text-gray-500">
                world {g.choice.world} · {formatBytes(g.choice.payload_bytes)}
              </span>
              <span
                className="ml-1.5 rounded-full px-1.5 font-mono text-10 font-medium"
                style={{ color: algorithmColor(g.choice.algorithm), background: "#f8fafc" }}
              >
                {g.choice.algorithm}
              </span>
              <span className="ml-1.5 font-mono text-gray-600">
                {formatTokens(g.launches)}× · {fmtTime(g.t_total_s)}
              </span>
            </button>
          );
        })}
      </div>
      <CommGroupPanel key={`${tab}-${group.key}`} group={group} phase={tab} />
    </section>
  );
}

function CommGroupPanel({ group, phase }: { group: CommGroup; phase: Phase }) {
  const choice = group.choice;
  const count = choice.count > 0 ? choice.count : 1;
  const chosen = chosenCandidate(choice);
  const feasible = feasibleCandidates(choice);
  const envelope = envelopeCurve(choice);
  const title = `${phase} ${collectiveLabel(choice.op)} world ${choice.world}`;
  // One launch's times: the backend prices `count` instances at once.
  const perLaunchCurve = (curve: [number, number][]) =>
    curve.map(([x, t]) => [x, t / count]);
  const bandSeries = envelope.length
    ? [{
      id: "envelope",
      name: "envelope = the fabric's best",
      type: "line" as const,
      data: envelope.map(([x, t, algorithm]) => ({ value: [x, t / count], name: algorithm })),
      symbol: "none" as const,
      silent: true,
      z: 1,
      lineStyle: { width: 7, color: "#94a3b8", opacity: 0.3 },
    }]
    : [];
  const option: EChartsOption = {
    grid: { left: 64, right: 16, top: 48, bottom: 44 },
    // The legend keeps clear of the toolbox in the top-right corner.
    legend: { type: "scroll", top: 0, left: 0, right: 100, textStyle: { fontSize: 9 }, itemWidth: 14 },
    tooltip: {
      trigger: "axis",
      confine: true,
      order: "valueAsc",
      axisPointer: { label: { formatter: ({ value }) => formatBytes(Number(value)) } },
      valueFormatter: (v) => fmtDuration(Number(v)),
    },
    xAxis: {
      type: "log",
      // Decades, not powers of two: the grid spans 1 KiB to 1 GiB, and
      // base-2 ticks there thin out to the two ends.
      logBase: 10,
      name: "payload per launch (bytes, log)",
      nameLocation: "middle",
      nameGap: 26,
      ...AXIS,
      axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => formatBytes(v) },
    },
    yAxis: {
      type: "log",
      name: "time per launch (log)",
      nameLocation: "middle",
      nameGap: 46,
      ...AXIS,
      axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => fmtDuration(v) },
    },
    series: [
      ...bandSeries,
      ...feasible.map((c, i) => {
        const won = c.algorithm === choice.algorithm;
        const colour = algorithmColor(c.algorithm, i);
        return {
          id: c.algorithm,
          name: c.algorithm,
          type: "line" as const,
          data: perLaunchCurve(c.curve),
          symbol: "none" as const,
          z: won ? 4 : 2,
          lineStyle: { color: colour, width: won ? 2.4 : 1.2, type: (won ? "solid" : "dashed") as "solid" | "dashed" },
          itemStyle: { color: colour },
        };
      }),
      {
        id: "here",
        name: "this collective",
        type: "scatter",
        data: chosen?.t_s != null ? [[choice.payload_bytes, chosen.t_s / count]] : [],
        symbolSize: 12,
        z: 6,
        itemStyle: { color: algorithmColor(choice.algorithm), borderColor: "#ffffff", borderWidth: 2 },
      },
    ],
    ...interactiveBits({ name: title }),
  };
  return (
    <div className="grid gap-4 lg:grid-cols-2" data-comm-panel={group.key}>
      <div>
        <div className="text-xs font-semibold text-gray-700 mb-1">
          {collectiveLabel(choice.op)} over {choice.world} ranks —{" "}
          <span className="font-mono" style={{ color: algorithmColor(choice.algorithm) }}>{choice.algorithm}</span>
        </div>
        <div className="text-10 text-gray-400 mb-2">{fabricLabel(choice.fabric)}</div>
        <CommTopology choice={choice} stages={chosen?.stages || []} />
        {chosen?.description ? (
          <div className="mt-1 text-11 text-gray-600" data-comm-caption="">{chosen.description}.</div>
        ) : null}
        <Numbers group={group} />
      </div>
      <div>
        <div className="text-xs font-semibold text-gray-700 mb-1">
          Every algorithm this fabric was offered, priced across payload size
        </div>
        <HaChart option={option} height={260} />
        <div className="text-10 text-gray-400 mt-1">
          {envelope.length
            ? "The thick band is the envelope the step is priced under — at each payload, the backend's own minimum over the algorithms the fabric can run. "
            : "This payload publishes no envelope line, so only the candidates' own curves are drawn. "}
          The dot is this collective&apos;s payload on the algorithm that won. Times are one launch&apos;s.
        </div>
        <div className="mt-2 space-y-1" data-comm-candidates="">
          {choice.candidates.map((c, i) => (
            <CandidateRow key={c.algorithm} candidate={c} index={i} chosen={c.algorithm === choice.algorithm} count={count} />
          ))}
        </div>
      </div>
    </div>
  );
}

/** One candidate as the backend published it: its name, its own sentence, its
 *  per-launch time, and either why the fabric refused it or the caveat the
 *  backend attached to its price. */
function CandidateRow({ candidate, index, chosen, count }: {
  candidate: CommCandidate;
  index: number;
  chosen: boolean;
  count: number;
}) {
  const t = perLaunch(candidate.t_s, count);
  const wire = perLaunch(candidate.wire_bytes_per_rank, count);
  return (
    <div
      className={`rounded-md border px-2 py-1 ${chosen ? "border-amber-300 bg-amber-50/50" : "border-gray-100"}`}
      data-comm-candidate={candidate.algorithm}
      data-feasible={candidate.feasible ? "1" : "0"}
    >
      <div className="flex items-baseline gap-1.5 text-11">
        <span className="inline-block h-2 w-2 shrink-0 rounded-sm" style={{ background: algorithmColor(candidate.algorithm, index) }} />
        <span className={`font-mono ${candidate.feasible ? "text-gray-800" : "text-gray-400 line-through"}`}>
          {candidate.algorithm}
        </span>
        {chosen ? <span className="rounded-full bg-amber-100 px-1.5 text-10 font-medium text-amber-800">chosen</span> : null}
        <span className="flex-1" />
        <span className="font-mono tabular-nums text-gray-600">
          {t != null ? fmtTime(t) : "—"}
          {wire != null ? ` · ${formatBytes(wire)}/rank` : ""}
          {candidate.hops != null ? ` · ${candidate.hops} hop${candidate.hops === 1 ? "" : "s"}` : ""}
        </span>
      </div>
      <div className="text-10 text-gray-500">{candidate.description}</div>
      {candidate.reason ? (
        <div className="text-10 text-gray-500" data-comm-reason="">not available here — {candidate.reason}</div>
      ) : null}
      {candidate.note ? (
        <div className="text-10 text-amber-700" data-comm-note="">{candidate.note}</div>
      ) : null}
    </div>
  );
}

/** One launch's figures and the group's roll-up. Every number is the
 *  backend's: the per-launch figures are the chosen candidate's own divided
 *  by the instances it was priced over, and the step total is the sum of the
 *  rows' published `t_comm` — never launches × one launch's time. */
function Numbers({ group }: { group: CommGroup }) {
  const choice = group.choice;
  const count = choice.count > 0 ? choice.count : 1;
  const chosen = chosenCandidate(choice);
  const wire = perLaunch(chosen?.wire_bytes_per_rank, count);
  const t = perLaunch(chosen?.t_s, count);
  const bw = group.bw_label
    ? group.bw_bytes_per_sec
      ? `${group.bw_label} · ${formatBytes(group.bw_bytes_per_sec)}/s`
      : group.bw_label
    : "—";
  const items: { label: string; value: ReactNode }[] = [
    { label: "Launches per step", value: formatTokens(group.launches) },
    { label: "Payload per launch", value: formatBytes(choice.payload_bytes) },
    { label: "Wire bytes / rank / launch", value: wire != null ? formatBytes(wire) : "—" },
    { label: "Bandwidth", value: bw },
    { label: "Hops per launch", value: chosen?.hops != null ? String(chosen.hops) : "—" },
    {
      label: "t per launch",
      value: t != null
        ? <FormulaHover value={fmtTime(t)} formula={group.formula ?? undefined} cls="text-sky-700" />
        : "—",
    },
    { label: "Σ over the step", value: fmtTime(group.t_total_s) },
  ];
  return (
    <dl className="mt-3 grid max-w-sm grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-11" data-comm-numbers="">
      {items.map((it) => (
        <Fragment key={it.label}>
          <dt className="text-gray-500">{it.label}</dt>
          <dd className="text-right font-mono tabular-nums text-gray-800">{it.value}</dd>
        </Fragment>
      ))}
      {group.paths.length > 0 && (
        <div className="col-span-2 mt-1 text-10 text-gray-400" data-comm-paths="">
          {group.paths.length === 1 ? "Launched at " : `Launched at ${group.paths.length} places, e.g. `}
          <span className="font-mono">{group.paths.slice(0, 3).join(", ")}</span>
          {group.paths.length > 3 ? ", …" : ""}
        </div>
      )}
    </dl>
  );
}
