"""`Layer[T]` — one derived fact about every node, and `Rewrite` — what a graph
transform did, in a form layers can follow.

Derived facts belong to distinct concerns: identity, cost, shape, axis labels,
symbolic expressions, display strings and resolved dimensions. Keeping those
concerns in one tree would make every consumer depend on every fact and force
graph rewrites to reconstruct them together.

A `Layer` is one of those concerns, on its own, keyed by node uid:

    shapes:   Layer[Shape]
    cost:     Layer[Cost]
    formulas: Layer[Expr]
    display:  formulas.zip(shapes).map(render)      # derived, not stored

`map` / `zip` make the derived ones derived rather than persisted, which is the
point: `display` costs nothing until someone renders, instead of ten strings
built per node per walk.

Keying on `uid` rather than on the composed `aggregation_id` string is the other
half. Several node identities are useful at boundaries — local name, uid,
tensor name, and the composed `f"{template}|{display_path}"` — and crossing
between representations means string surgery. `SpecIRTree` already allocates a
tree-unique uid per node and hands out fresh ones for nodes created during
mutation, so the identity exists; it just isn't what anything joins on.

Layers do not all come from the walk
------------------------------------
`Shapes` / `Cost` / `Formulas` / `Placement` propagate along edges. `Formats`
(quant) does not: a weight's dtype follows from its role and the model's
`QuantContext`, not from what feeds it. `Measured` does not either — it is
external corpus joined in by identity. Those two are sideband layers, aligned by
uid and never derived by traversal, which is the structural reading of "measured
sits beside the theory, it does not overwrite it".

`Bytes = Shapes.zip(Formats)` is where a propagated layer and a sideband layer
meet, and it is deliberately ONE place. Every consumer that re-derives this
join by name instead is a fork of the same fact — the failure mode is a read
path assuming a uniform fp8 while the storage path uses the native bf16 for
the same tensor, with nothing loud in between.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Generic, Iterator, Mapping, Optional,
    Sequence, Tuple, TypeVar,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .node import Identified
    from .program import Program

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
K = TypeVar("K")

#: Distinct from a key whose value is legitimately `None`, which is a bucket.
_MISSING = object()


class MissingValue(KeyError):
    """A layer was asked for a node it has no value for.

    Distinct from a value that is legitimately `None`: a layer may hold None for
    a node whose shape could not be inferred, and that is an answer. Absence is
    not, so it raises rather than returning None and letting the caller mistake
    one for the other.
    """


@dataclass(frozen=True)
class Rewrite:
    """What a `GraphTransform` did, as a mapping layers can be carried across.

    A graph transform is not only a new graph — it is also the correspondence
    between the old nodes and the new ones, and without that correspondence
    every existing layer silently becomes garbage: its keys refer to nodes the
    new graph no longer contains, and a lookup that should fail instead misses
    into whatever uid happens to be reused.

    `origins` maps each NEW node to the old nodes it came from: three
    projections fused into one QKV node map that node to those three. A node
    the transform left alone does not appear (identity is implied), and a node
    it deleted appears nowhere.

    Storage is deliberately out of scope. Fusion rewrites the COMPUTE graph —
    a fused QKV still reads three weight tensors, each with its own dtype and
    its own bytes on disk — so the sideband `Formats` layer keys on storage
    nodes that no compute-graph rewrite touches, and needs no rebase at all.
    Giving `Rewrite` a "preserve" mode for it would invent a second, subtly
    different meaning for the same operation.
    """

    origins: Mapping[int, Tuple[int, ...]] = field(default_factory=dict)
    dropped: frozenset = frozenset()
    #: Each input of a NEW node, as the input of the old node it carries:
    #: `(new uid, input) -> (old uid, input)`. A fused node reads, at each
    #: input, what the member whose edge it took over read there.
    inputs: Mapping[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict)
    #: Which runs of each old node each NEW node took: `new uid -> {old
    #: uid -> {iteration of the new node -> iteration of the old one}}`.
    #: A launch within one iteration takes the same iteration of every
    #: member; one across a layer seam takes the reading iteration of the
    #: reading layer's members and the writing iteration of the previous
    #: layer's. The new node's own iterations are its home's keys. An old
    #: node with runs no new node took keeps them as a node of its own (not
    #: `dropped`) — the norm that opens a stack keeps its layer-0 run. Every
    #: count (`Layer.rebase_counts`) and every "which launch writes this
    #: tensor at this iteration" (`launch_at`) is read off this one map.
    #: Absent for a rewrite with no walk behind it: then a new node took
    #: every run of each origin.
    taken: Mapping[int, Mapping[int, Mapping[Optional[int], Optional[int]]]] \
        = field(default_factory=dict)
    #: Each new node's home: the old node whose iterations its own index
    #: (`FusedNode.home`). An old node is its own.
    homes: Mapping[int, int] = field(default_factory=dict)

    @property
    def added(self) -> frozenset:
        return frozenset(self.origins)

    def sources(self, uid: int) -> Tuple[int, ...]:
        """The old nodes `uid` was built from; empty for an untouched node."""
        return tuple(self.origins.get(uid, ()))

    def is_identity(self) -> bool:
        return (not self.origins and not self.dropped and not self.inputs
                and not self.taken)

    def home(self, uid: int) -> int:
        """The old node whose iterations `uid`'s own index."""
        return self.homes.get(uid, uid)

    def whole(self, old_uid: int) -> bool:
        """Whether new nodes took every run of `old_uid` (it is gone)."""
        return old_uid in self.dropped

    def partly(self, old_uid: int) -> bool:
        """Whether a new node took some runs of `old_uid` and it keeps the
        rest as a node of its own. In a composed rewrite (`then`) the old
        nodes are the first graph's: a launch an intermediate rewrite made
        and a later one took in part is a NEW node of the composition, its
        own runs restricted to the ones it kept, and not asked here."""
        return old_uid not in self.dropped and old_uid in self._taken_olds

    @property
    def _taken_olds(self) -> FrozenSet[int]:
        """Every old node some new node took a run of, built once."""
        cached = self.__dict__.get("_taken_olds_cache")
        if cached is None:
            cached = frozenset(old for olds in self.taken.values()
                               for old in olds)
            object.__setattr__(self, "_taken_olds_cache", cached)
        return cached

    def taken_by(self, old_uid: int) -> Tuple[int, ...]:
        """The new nodes that took runs of `old_uid`."""
        return tuple(new for new, olds in self.origins.items()
                     if old_uid in olds)

    def iterations(self, new_uid: int) -> FrozenSet[Optional[int]]:
        """The iterations a new node runs at, in its home's space."""
        return frozenset(self.taken[new_uid][self.home(new_uid)])

    def taken_at(self, old_uid: int) -> FrozenSet[Optional[int]]:
        """The iterations of `old_uid` new nodes took."""
        return frozenset(it for olds in self.taken.values()
                         for it in olds.get(old_uid, {}).values())

    def launch_at(self, old_uid: int, iteration: Optional[int]
                  ) -> Optional[int]:
        """The new node that runs `old_uid`'s iteration `iteration` — which
        launch writes that run's tensor — None when no new node took it.
        Without a walk behind the rewrite, the one that took the node."""
        if not self.taken:
            found = [new for new, olds in self.origins.items()
                     if old_uid in olds and old_uid in self.dropped]
            return found[0] if found else None
        return self._run_by.get((old_uid, iteration))

    def iteration_classes(self, program: Any, block: int
                          ) -> Tuple[FrozenSet[Optional[int]], ...]:
        """The iterations of block invocation `block` grouped by what runs
        there: two iterations are one class when every node of the block a
        launch took a run of is run by the same launch at both (or by none
        at both). A template's layers can differ in this — DeepSeek-V3's
        first MoE layer opens with the dense → MoE seam, the others with
        the MoE → MoE one, and the last closes into the final norm — and a
        consumer that simulates or prices one layer does so once per class
        (`activation_peak`, `traffic.derive`). Read off `taken` alone; one
        class, the whole block, where no launch took a run of it."""
        # Kept beside the program it answers for: an entry holds its
        # program, so a later program at a reused address is no hit.
        cache = self.__dict__.setdefault("_classes_cache", {})
        held = cache.get(id(program))
        if held is None or held[0] is not program:
            held = cache[id(program)] = (program, {})
        cache = held[1]
        key = block
        if key not in cache:
            iterations = program.blocks[block].layer_indices
            members = sorted({old for olds in self.taken.values()
                              for old in olds
                              if program.block_of(old) == block})
            groups: Dict[Tuple[Any, ...], list] = {}
            for it in iterations:
                groups.setdefault(tuple(
                    self.launch_at(member, it) for member in members),
                    []).append(it)
            cache[key] = tuple(frozenset(its) for its in groups.values())
        return cache[key]

    @property
    def _run_by(self) -> Mapping[Tuple[int, Any], int]:
        """`(old uid, iteration) -> new uid`, built once: one run is one
        launch."""
        cached = self.__dict__.get("_run_by_cache")
        if cached is None:
            cached = {}
            for new, olds in self.taken.items():
                for old, at in olds.items():
                    for it in at.values():
                        other = cached.setdefault((old, it), new)
                        if other != new:
                            raise ValueError(
                                f"iteration {it!r} of uid {old} is run by "
                                f"{sorted((other, new))!r}; one run is one "
                                f"launch")
            object.__setattr__(self, "_run_by_cache", cached)
        return cached

    def then(self, other: "Rewrite") -> "Rewrite":
        """Compose two rewrites applied in order, flattening origins through the
        intermediate graph so a layer can cross both in one `rebase`."""
        origins: Dict[int, Tuple[int, ...]] = {}
        for new_uid, mid_uids in other.origins.items():
            flat: list = []
            for mid in mid_uids:
                first = self.origins.get(mid)
                flat.extend(first if first is not None else (mid,))
            origins[new_uid] = tuple(flat)
        for new_uid, srcs in self.origins.items():
            if new_uid not in origins and new_uid not in other.dropped:
                origins[new_uid] = srcs
        inputs: Dict[Tuple[int, int], Tuple[int, int]] = {
            key: self.inputs.get(mid, mid)
            for key, mid in other.inputs.items()}
        for key, old in self.inputs.items():
            if key not in inputs and key[0] not in other.dropped:
                inputs[key] = old
        # A node made from another takes runs of that one's origins: the
        # iteration of the old node behind each of its own (`taken`).
        homes: Dict[int, int] = {
            new: self.home(mid) for new, mid in other.homes.items()}
        for new, home in self.homes.items():
            if new in origins and new not in homes:
                homes[new] = home
        if other.taken and not self.taken and self.origins:
            # Which runs of the first graph's nodes the intermediate one's
            # stand for is unstated: composing would name an intermediate
            # node as an old one.
            raise ValueError(
                "a rewrite that states its runs composes only after one "
                "that states its own")
        taken: Dict[int, Dict[int, Dict[Any, Any]]] = {}
        for new, mids in other.taken.items():
            flat: Dict[int, Dict[Any, Any]] = {}
            for mid, at in mids.items():
                for old, behind in self.taken.get(mid, {mid: None}).items():
                    flat.setdefault(old, {}).update(
                        at if behind is None else
                        {it: behind[mid_it] for it, mid_it in at.items()
                         if mid_it in behind})
            taken[new] = flat
        for new, olds in self.taken.items():
            if new in other.dropped:
                continue
            # A node of this rewrite a later one took in part keeps the runs
            # the later one did not take.
            kept = (frozenset(olds[self.home(new)])
                    - other.taken_at(new))
            taken[new] = {old: {it: at[it] for it in at if it in kept}
                          for old, at in olds.items()}
        return Rewrite(origins=origins, dropped=self.dropped | other.dropped,
                       inputs=inputs, taken=taken, homes=homes)


@dataclass(frozen=True)
class Layer(Generic[T]):
    """One derived fact per node, keyed by node uid.

    `program` is the graph the keys refer to, and it is optional for one reason:
    a layer built mid-walk has values before it has anywhere to look a uid up.
    Everything a finished `Analysis` hands out carries it, so a derivation that
    needs the node — an op name, a role — asks the layer rather than being
    handed the program a second time alongside it.
    """

    program: Optional["Program"] = None
    values: Mapping[int, T] = field(default_factory=dict)

    # -- reading ---------------------------------------------------------

    def __getitem__(self, node: Any) -> T:
        uid = _uid_of(node)
        try:
            return self.values[uid]
        except KeyError:
            raise MissingValue(
                f"layer has no value for uid {uid}"
                f"{f' ({node.name!r})' if hasattr(node, 'name') else ''}"
            ) from None

    def get(self, node: Any, default: Any = None) -> Any:
        return self.values.get(_uid_of(node), default)

    def __contains__(self, node: Any) -> bool:
        return _uid_of(node) in self.values

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)

    # -- deriving --------------------------------------------------------

    def map(self, fn: Callable[[T], U]) -> "Layer[U]":
        """Apply `fn` to every value. The derived-not-stored path: `display` is
        `formulas.zip(shapes).map(render)`, computed when read."""
        return Layer(self.program, {u: fn(v) for u, v in self.values.items()})

    def map_with_node(self, fn: Callable[["Identified", T], U]) -> "Layer[U]":
        """`map`, for derivations that also need the node (an op name, a role)."""
        if self.program is None:
            raise ValueError("map_with_node needs a program to resolve uids")
        by_uid = self.program.by_uid
        return Layer(self.program,
                     {u: fn(by_uid(u), v) for u, v in self.values.items()})

    def filter(self, pred: Callable[[T], bool]) -> "Layer[T]":
        return Layer(self.program,
                     {u: v for u, v in self.values.items() if pred(v)})

    def leaves(self) -> "Layer[T]":
        """Only the nodes with nothing beneath them.

        The restriction every aggregation over a TREE needs and none of them
        should spell out: a component's value already includes its children, so
        summing both counts the subtree twice. Stated once here rather than as a
        `if node.children: continue` inside each consumer's loop, which is where
        it was and which is a filter wearing a traversal's clothes.
        """
        if self.program is None:
            raise ValueError("leaves needs a program to know what has children")
        by_uid = self.program.by_uid
        return Layer(self.program, {u: v for u, v in self.values.items()
                                    if not by_uid(u).children})

    def group_by(self, keys: "Layer[K]",
                 combine: Callable[[Sequence[T]], T]) -> Dict[K, T]:
        """Fold this layer's values into buckets named by another layer.

        Attribution — cost by role, cost by stack — is a join and a group-by
        once both sides are layers. The engine reaches the same answer with a
        recursive classifier that re-descends the result tree deciding per node
        which bucket it belongs to, and a node it cannot classify falls through
        a chain of tests into "neither"; here a node with no key is simply not
        in the key layer, which is the same statement without the fall-through.

        The intersection of the two domains, for the reason `zip` gives: a key
        layer is partial by construction — only a roled node has a role.
        """
        buckets: Dict[K, list] = {}
        for uid, value in self.values.items():
            key = keys.values.get(uid, _MISSING)
            if key is _MISSING:
                continue
            buckets.setdefault(key, []).append(value)  # type: ignore[arg-type]
        return {key: combine(parts) for key, parts in buckets.items()}

    def zip(self, other: "Layer[U]") -> "Layer[Tuple[T, U]]":
        """Pair values on the nodes BOTH layers cover.

        The intersection is the honest join: a propagated layer covers every
        node while a sideband layer covers only the ones it has data for
        (`Measured` over a partial corpus, `Formats` over weight-bearing nodes),
        so requiring equal domains would make the common case an error.
        """
        return Layer(self.program or other.program,
                     {u: (v, other.values[u])
                      for u, v in self.values.items() if u in other.values})

    def zip_with(self, other: "Layer[U]",
                 fn: Callable[[T, U], V]) -> "Layer[V]":
        return self.zip(other).map(lambda pair: fn(pair[0], pair[1]))

    # -- following a graph transform -------------------------------------

    def rebase(self, rw: Rewrite, program: Optional["Program"],
               combine: Callable[[Tuple[T, ...]], T]) -> "Layer[T]":
        """Carry this layer across a graph transform.

        A node the transform left alone keeps its value. A node it created takes
        `combine(values of its origins)` — for cost that is a sum, for shape it
        is the surviving endpoint's. A node it dropped is gone.

        `combine` is required rather than defaulted because there is no correct
        default: silently taking the first origin's value would make a fused
        QKV report one projection's FLOPs, which is exactly the kind of
        three-times-under-count that reads as plausible.
        """
        out: Dict[int, T] = {u: v for u, v in self.values.items()
                             if u not in rw.dropped and u not in rw.origins}
        for new_uid, srcs in rw.origins.items():
            present = tuple(self.values[s] for s in srcs if s in self.values)
            if present:
                out[new_uid] = combine(present)
        return Layer(program if program is not None else self.program, out)

    def rebase_counts(self, rw: Rewrite, program: Optional["Program"],
                      held: Optional[Callable[[int, Optional[int]], bool]]
                      = None) -> "Layer[T]":
        """Carry a COUNT layer — how many times each node runs — across a
        transform, read off the one statement of which runs each node took
        (`Rewrite.taken`): a new node runs its home's count per iteration
        times its iterations, and an old node it took some runs of keeps the
        rest. `held(uid, iteration)` says which iterations this
        layer counts (one pipeline stage's), all of them when None.

        A count is a number on this path: a symbolic count has no runs to
        take a share of, and is refused rather than branched on."""
        source = self.program

        def runs_of(value: Any, uid: int) -> int:
            try:
                count = int(value)
            except TypeError:
                raise ValueError(
                    f"uid {uid}: a count crosses a rewrite only as a number "
                    f"of runs, not {value!r}") from None
            if count != value:
                raise ValueError(
                    f"uid {uid}: {value!r} is no whole number of runs")
            return count

        def counted(uid: int, its: Any) -> list:
            return [it for it in its if held is None or held(uid, it)]

        def per_run(uid: int) -> Optional[int]:
            counted_runs = counted(uid, source.occurrence_layers(uid))
            count = runs_of(self.values[uid], uid)
            if not counted_runs:
                if count:
                    raise ValueError(
                        f"uid {uid} runs {count} times at no iteration")
                return None
            if count % len(counted_runs):
                raise ValueError(
                    f"uid {uid}: {count} runs over {len(counted_runs)} "
                    f"iterations is no whole count per iteration")
            return count // len(counted_runs)

        def count_of(home: int, its: Any) -> int:
            each = per_run(home)
            return 0 if each is None else each * len(counted(home, its))

        out: Dict[int, T] = {}
        for uid, value in self.values.items():
            if uid in rw.dropped or uid in rw.origins:
                continue
            out[uid] = (count_of(uid, frozenset(
                source.occurrence_layers(uid)) - rw.taken_at(uid))
                if rw.partly(uid) else value)
        for new_uid, srcs in rw.origins.items():
            present = [s for s in srcs if s in self.values]
            if not present:
                continue
            if new_uid not in rw.taken:
                counts = {self.values[s] for s in present}
                if len(counts) > 1:
                    raise ValueError(
                        f"fusion spans unequal per-step facts: uid "
                        f"{new_uid} runs {sorted(counts)} times by its "
                        f"origins")
                out[new_uid] = counts.pop()
                continue
            # A launch runs once per run of its home; every origin it took
            # runs as often per iteration, or the launch is no one kernel.
            each = {per_run(s) for s in present} - {None}
            if len(each) > 1:
                raise ValueError(
                    f"fusion spans unequal per-step facts: uid {new_uid}'s "
                    f"origins run {sorted(each)} times per iteration")
            out[new_uid] = count_of(rw.home(new_uid),
                                    rw.iterations(new_uid))
        return Layer(program if program is not None else self.program, out)

    # -- building --------------------------------------------------------

    @classmethod
    def build(cls, program: Optional["Program"], pairs) -> "Layer[T]":
        """From `(node_or_uid, value)` pairs."""
        return cls(program, {_uid_of(k): v for k, v in pairs})

    def with_value(self, node: Any, value: T) -> "Layer[T]":
        merged = dict(self.values)
        merged[_uid_of(node)] = value
        return Layer(self.program, merged)

    def __repr__(self) -> str:
        return f"Layer(n={len(self.values)})"


def _uid_of(node: Any) -> int:
    """Accept a node or a bare uid. Layers are keyed by uid, but callers holding
    a node should not have to reach for `.uid` at every call site."""
    if isinstance(node, int):
        return node
    uid = getattr(node, "uid", None)
    if uid is None:
        raise TypeError(f"not a node and not a uid: {node!r}")
    return uid
