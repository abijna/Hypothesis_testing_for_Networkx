# Project: Property-Based Testing for NetworkX
#
# SR No: 13-19-01-19-52-24-1-24725
# Team Members: 1
# Team Member Name: Abijna Rao
# Library: NetworkX 3.6.1, Hypothesis 6.x, pytest
#
# Algorithms tested in this file:
#   - Shortest paths: Dijkstra, Bellman-Ford, bidirectional Dijkstra,
#     Johnson's all-pairs algorithm
#   - Minimum spanning tree
#   - Group betweenness centrality
#
# Recommended run command:
#   pytest -v networkx_properties_tests.py
# Alternative:
#   python3 networkx_properties_tests.py
#
"""Property-based tests for a selected subset of NetworkX graph algorithms."""

import math
import random

import networkx as nx
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st


@st.composite
def _int_weighted_graph(draw, directed=False, min_nodes=2, max_nodes=10,
                        min_weight=1, max_weight=10, ensure_connected=False):
    """Generate a random graph with integer edge weights."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    p = draw(st.floats(min_value=0.15, max_value=0.85))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    G = nx.gnp_random_graph(n, p, seed=seed, directed=directed)

    if ensure_connected and n > 1:
        # When connectivity is required, a simple chain is added through a
        # shuffled node order. This keeps the generator flexible while still
        # guaranteeing that the shortest-path and MST tests have meaningful
        # connected inputs to work with.
        rng = random.Random(seed)
        order = list(G.nodes())
        rng.shuffle(order)
        for u, v in zip(order, order[1:]):
            G.add_edge(u, v)
            if directed:
                G.add_edge(v, u)

    # Each edge receives its weight from Hypothesis so that structure and
    # weight assignment are explored independently.
    for u, v in G.edges():
        w = draw(st.integers(min_value=min_weight, max_value=max_weight))
        G[u][v]["weight"] = w
    return G


@st.composite
def _float_weighted_connected_graph(draw, min_nodes=3, max_nodes=8):
    """Generate a connected undirected graph with positive float weights."""
    G = draw(_int_weighted_graph(min_nodes=min_nodes, max_nodes=max_nodes,
                                 ensure_connected=True))
    for u, v in G.edges():
        G[u][v]["weight"] = draw(st.floats(min_value=0.1, max_value=50.0,
                                           allow_nan=False,
                                           allow_infinity=False))
    return G


@st.composite
def _graph_with_two_reachable_nodes(draw, directed=False):
    """Generate a graph together with two distinct nodes joined by a path."""
    G = draw(_int_weighted_graph(directed=directed, ensure_connected=True,
                                 min_nodes=3, max_nodes=10))
    nodes = list(G.nodes())
    u = draw(st.sampled_from(nodes))
    v = draw(st.sampled_from([n for n in nodes if n != u]))
    assume(nx.has_path(G, u, v))
    return G, u, v


@st.composite
def _diverse_topology(draw, min_n=4, max_n=10):
    """
    Generate a graph from a family of distinct topologies, then assign integer
    weights in [1, 10]. The supported families are:

        gnp        -- Erdos-Renyi random
        complete   -- K_n
        cycle      -- C_n
        path       -- P_n
        wheel      -- W_n (hub + cycle)
        ba         -- Barabasi-Albert preferential attachment (scale-free)
        regular    -- random d-regular

    The purpose of this strategy is not only variety for its own sake. Some
    graph bugs appear only on particular structures, so it is helpful to let
    Hypothesis move across qualitatively different families rather than remain
    inside a single random graph model.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    kind = draw(st.sampled_from(
        ["gnp", "complete", "cycle", "path", "wheel", "ba", "regular"]
    ))
    if kind == "gnp":
        p = draw(st.floats(min_value=0.25, max_value=0.85))
        G = nx.gnp_random_graph(n, p, seed=seed)
    elif kind == "complete":
        G = nx.complete_graph(n)
    elif kind == "cycle":
        G = nx.cycle_graph(n)
    elif kind == "path":
        G = nx.path_graph(n)
    elif kind == "wheel":
        G = nx.wheel_graph(n)
    elif kind == "ba":
        m = draw(st.integers(min_value=1, max_value=max(1, n // 2)))
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    else:
        d = draw(st.integers(min_value=2, max_value=max(2, n - 1)))
        if (d * n) % 2 != 0:
            d -= 1
        if d < 2 or d >= n:
            G = nx.cycle_graph(n)
        else:
            try:
                G = nx.random_regular_graph(d, n, seed=seed)
            except nx.NetworkXError:
                G = nx.cycle_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = draw(st.integers(min_value=1, max_value=10))
    return G


_PBT_SETTINGS = settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much,
                           HealthCheck.data_too_large],
)

def _path_weight(G, path, weight="weight"):
    """Return the total weight of a node sequence interpreted as a path."""
    return sum(G[u][v][weight] for u, v in zip(path, path[1:]))


# ------------------------------------------------------------------------------
#                               POSTCONDITIONS
# ------------------------------------------------------------------------------


@_PBT_SETTINGS
@given(_graph_with_two_reachable_nodes())
def test_dijkstra_path_is_valid_walk_with_matching_length(payload):
    """
    Property
    --------
    `dijkstra_path(G, u, v)` returns a node sequence P such that:
      (a) P[0] == u, P[-1] == v;
      (b) every consecutive pair (P[i], P[i+1]) is a real edge of G;
      (c) the summed edge weight along P equals `dijkstra_path_length(G, u, v)`.

    Mathematical basis
    ------------------
    `dijkstra_path` and `dijkstra_path_length` are two separately-implemented
    public functions, so they ought to agree by construction. In addition, the
    returned node sequence should genuinely describe a walk in G rather than a
    merely plausible answer.

    Test strategy
    -------------
    A connected weighted graph together with two reachable nodes is generated,
    and the three structural conditions above are then verified directly.

    Assumptions / Preconditions
    ---------------------------
    Edge weights are positive, and the helper strategy guarantees that the
    chosen endpoints u and v are connected by at least one path.

    Why this matters
    ----------------
    This test checks that the path-producing and length-producing interfaces
    agree with one another and with the actual graph structure. A failure would
    indicate a bug in path reconstruction, edge validation, or weight
    accounting inside the shortest-path implementation.
    """
    G, u, v = payload
    path = nx.dijkstra_path(G, u, v)
    assert path[0] == u and path[-1] == v
    for a, b in zip(path, path[1:]):
        assert G.has_edge(a, b), f"Edge {(a, b)} not in G"
    assert math.isclose(_path_weight(G, path),
                        nx.dijkstra_path_length(G, u, v),
                        rel_tol=1e-9, abs_tol=1e-9)


# ------------------------------------------------------------------------------
#                            METAMORPHIC PROPERTIES
# ------------------------------------------------------------------------------


@_PBT_SETTINGS
@given(_float_weighted_connected_graph(),
       st.floats(min_value=0.01, max_value=1000.0, allow_nan=False,
                 allow_infinity=False))
def test_dijkstra_positive_weight_scaling(G, c):
    """
    Property
    --------
    For any positive constant c > 0 and any source s,
        d_{cG}(s, t) == c * d_G(s, t)
    for every reachable t, where cG denotes G with every edge weight
    multiplied by c.

    Mathematical basis
    ------------------
    Multiplying every weight by c > 0 multiplies the cost of every walk by c
    while preserving the ordering of walks by total cost. The shortest path
    should therefore remain shortest, with its value scaled by the same factor.

    Test strategy
    -------------
    A connected float-weighted graph is generated, every edge is scaled by c,
    and the resulting single-source distance dictionaries are compared from the
    same source.

    Assumptions / Preconditions
    ---------------------------
    The graph is connected, all edge weights are strictly positive, and the
    scaling constant c is also strictly positive.

    Why this matters
    ----------------
    Dijkstra's algorithm should depend on the ordering of path costs rather
    than their absolute magnitude. A failure here would suggest a numerical
    error, incorrect weight handling, or some unintended dependence on raw
    weight scale.
    """
    source = next(iter(G.nodes()))
    G2 = G.copy()
    for u, v in G2.edges():
        G2[u][v]["weight"] *= c
    d1 = nx.single_source_dijkstra_path_length(G, source)
    d2 = nx.single_source_dijkstra_path_length(G2, source)
    assert d1.keys() == d2.keys()
    for t in d1:
        assert math.isclose(d2[t], c * d1[t], rel_tol=1e-7, abs_tol=1e-7)


@_PBT_SETTINGS
@given(_int_weighted_graph(ensure_connected=True))
def test_dijkstra_equals_bellman_ford_on_nonneg_weights(G):
    """
    Property
    --------
    On graphs with non-negative weights, Dijkstra and Bellman-Ford must return
    identical single-source distance maps from every source.

    Mathematical basis
    ------------------
    Both algorithms compute the same shortest-path distance function. They
    differ in efficiency and in Bellman-Ford's support for negative weights,
    but on non-negative inputs their answers should coincide exactly.

    Test strategy
    -------------
    A connected integer-weighted graph (weights >= 1) is generated and
    `single_source_dijkstra_path_length` is compared with
    `single_source_bellman_ford_path_length` for every source.

    Assumptions / Preconditions
    ---------------------------
    All generated edge weights are non-negative, which places the input in the
    regime where both algorithms are expected to compute the same distances.

    Why this matters
    ----------------
    This is a cross-validation test between two independent shortest-path
    routines. If the distance maps differ, at least one of the implementations
    is incorrect, which strongly suggests a genuine algorithmic bug.
    """
    for source in G.nodes():
        d_dij = nx.single_source_dijkstra_path_length(G, source)
        d_bf = nx.single_source_bellman_ford_path_length(G, source)
        assert d_dij == d_bf


# ------------------------------------------------------------------------------
#                                 OPTIMALITY
# ------------------------------------------------------------------------------


@_PBT_SETTINGS
@given(_int_weighted_graph(ensure_connected=True, min_nodes=3, max_nodes=10))
def test_mst_weight_le_any_spanning_tree(G):
    """
    Property
    --------
    For a connected weighted graph G, the MST weight is a lower bound on the
    weight of any spanning tree of G.

    Mathematical basis
    ------------------
    The MST is, by definition, a minimum-weight spanning tree, so every other
    spanning tree must be at least as heavy.

    Test strategy
    -------------
    The MST weight is compared against the weight of a BFS spanning tree. BFS
    ignores edge weights, so it offers a convenient and intentionally naive
    reference tree.

    Assumptions / Preconditions
    ---------------------------
    The graph is connected and has positive integer edge weights, so both an
    MST and a BFS spanning tree are well-defined on the same vertex set.

    Why this matters
    ----------------
    An MST routine should never return a spanning tree heavier than an
    arbitrary alternative. A failure would indicate that the algorithm has
    chosen a non-optimal edge somewhere, which is precisely the kind of bug an
    optimality test is meant to reveal.
    """
    mst_weight = nx.minimum_spanning_tree(G).size(weight="weight")
    root = next(iter(G.nodes()))
    bfs = nx.bfs_tree(G, root).to_undirected()
    bfs_weight = sum(G[u][v]["weight"] for u, v in bfs.edges())
    assert mst_weight <= bfs_weight


# ------------------------------------------------------------------------------
#                                 IDEMPOTENCE
# ------------------------------------------------------------------------------


@_PBT_SETTINGS
@given(_int_weighted_graph(min_nodes=2, max_nodes=10))
def test_mst_idempotent(G):
    """
    Property
    --------
    `minimum_spanning_tree` is idempotent on its own output:

        MST(MST(G)) == MST(G)

    Mathematical basis
    ------------------
    Once a graph is already a spanning forest, re-running the MST routine
    should not change its edge set or its total weight. In other words, the
    output ought to be a fixed point of the algorithm.

    Test strategy
    -------------
    T = MST(G) is computed, followed by T2 = MST(T), and both the edge sets
    and total weights are compared.

    Assumptions / Preconditions
    ---------------------------
    The graph may be connected or disconnected. In either case,
    `minimum_spanning_tree` returns a spanning forest, and applying the same
    routine again should leave that forest unchanged.

    Why this matters
    ----------------
    Once an MST or minimum spanning forest has been produced, running the
    algorithm again should not alter the result. A failure would suggest
    instability, unnecessary rewiring, or an incorrect treatment of already
    optimal tree structure.
    """
    T = nx.minimum_spanning_tree(G)
    T2 = nx.minimum_spanning_tree(T)
    edges_T = {frozenset(e) for e in T.edges()}
    edges_T2 = {frozenset(e) for e in T2.edges()}
    assert edges_T == edges_T2
    assert T.size(weight="weight") == T2.size(weight="weight")


# ------------------------------------------------------------------------------
#                ADVANCED PROPERTIES (predecessor tree)
# ------------------------------------------------------------------------------


@_PBT_SETTINGS
@given(_diverse_topology(min_n=3, max_n=10))
def test_dijkstra_predecessor_tree(G):
    """
    Property
    --------
    `dijkstra_predecessor_and_distance(G, s)` returns predecessor information
    that can be turned into a shortest-path tree rooted at s.

    Mathematical basis
    ------------------
    In a graph with non-negative weights, shortest paths from a fixed source
    induce a tree on the reachable nodes where every tree edge satisfies
    dist[v] = dist[parent(v)] + weight(parent(v), v).

    Test strategy
    -------------
    A connected graph is generated, a source is chosen, one predecessor per
    node is selected, and edge validity, edge count, and distance
    compatibility are then verified.

    Assumptions / Preconditions
    ---------------------------
    The generated graph is required to be connected before the check proceeds,
    and all edge weights are positive integers so that shortest paths are
    well-defined for Dijkstra's algorithm.

    Why this matters
    ----------------
    Distance values alone do not guarantee that predecessor bookkeeping is
    correct. A failure here would point to an error in parent tracking or in
    the relationship between reported distances and the implied shortest-path
    tree.
    """
    assume(nx.is_connected(G))
    source = next(iter(G.nodes()))
    pred, dist = nx.dijkstra_predecessor_and_distance(G, source)
    tree_edges = [(preds[0], v) for v, preds in pred.items()
                  if v != source and preds]
    assert len(tree_edges) == len(dist) - 1
    for u, v in tree_edges:
        assert G.has_edge(u, v)
        assert math.isclose(dist[v], dist[u] + G[u][v]["weight"],
                            rel_tol=1e-9, abs_tol=1e-9)
    T = nx.Graph()
    T.add_nodes_from(dist.keys())
    T.add_edges_from(tree_edges)
    assert nx.is_tree(T)


# ------------------------------------------------------------------------------
#               CROSS-VALIDATION / ADVERSARIAL TESTS
# ------------------------------------------------------------------------------


@_PBT_SETTINGS
@given(_graph_with_two_reachable_nodes())
def test_bidirectional_dijkstra_matches_dijkstra(payload):
    """
    Property
    --------
    `bidirectional_dijkstra(G, u, v)` and `dijkstra_path_length(G, u, v)` must
    agree on the shortest-path length, and the path returned by the
    bidirectional variant must be a valid walk of exactly that length.

    Mathematical basis
    ------------------
    Bidirectional Dijkstra is an optimisation of the same shortest-path
    problem, so it must compute the same answer as forward Dijkstra.

    Test strategy
    -------------
    A connected graph and reachable endpoints are selected, the two reported
    lengths are compared, and the returned path is re-summed to confirm
    consistency.

    Assumptions / Preconditions
    ---------------------------
    The helper strategy ensures a connected graph with positive weights and two
    endpoints for which a path exists.

    Why this matters
    ----------------
    Bidirectional search is more subtle than standard forward Dijkstra because
    its stopping condition depends on two interacting search frontiers. A
    failure would indicate a bug in the bidirectional implementation even if
    the ordinary Dijkstra routine remains correct.
    """
    G, u, v = payload
    d_fwd = nx.dijkstra_path_length(G, u, v)
    d_bi, path = nx.bidirectional_dijkstra(G, u, v)
    assert d_fwd == d_bi
    assert path[0] == u and path[-1] == v
    assert _path_weight(G, path) == d_bi


@_PBT_SETTINGS
@given(_int_weighted_graph(ensure_connected=True, min_nodes=3, max_nodes=7))
def test_johnson_matches_all_pairs_dijkstra(G):
    """
    Property
    --------
    On graphs with non-negative edge weights, `johnson(G)` and
    `all_pairs_dijkstra_path_length(G)` must produce identical all-pairs
    shortest-path lengths.

    Mathematical basis
    ------------------
    Johnson's algorithm reweights edges and then runs Dijkstra from each
    source; on already non-negative graphs, the resulting distances must match
    plain all-pairs Dijkstra.

    Test strategy
    -------------
    The graph is converted to a DiGraph, Johnson paths and all-pairs Dijkstra
    lengths are computed, and the two are compared by summing the Johnson
    paths' edge weights.

    Assumptions / Preconditions
    ---------------------------
    The generated graph is connected and all weights are non-negative. After
    conversion to a directed graph, the same weight assumptions continue to
    hold, so Johnson's algorithm should agree with all-pairs Dijkstra.

    Why this matters
    ----------------
    Johnson's algorithm is a layered procedure that combines reweighting with
    repeated shortest-path computation. A failure would suggest an error in the
    composition itself, such as incorrect path reconstruction or incorrect
    interpretation of reweighted distances.
    """
    D = G.to_directed()
    apsp = dict(nx.all_pairs_dijkstra_path_length(D))
    j_paths = nx.johnson(D)
    for u in D.nodes():
        for v, path in j_paths[u].items():
            j_len = _path_weight(D, path)
            assert math.isclose(j_len, apsp[u][v],
                                rel_tol=1e-9, abs_tol=1e-9)


@_PBT_SETTINGS
@given(_int_weighted_graph(directed=True, ensure_connected=True,
                           min_nodes=3, max_nodes=8))
def test_dijkstra_transposition_isomorphism(G):
    """
    Property
    --------
    For any directed weighted graph G and any reachable pair (u, v),
        d_G(u, v) == d_{G^T}(v, u),
    where G^T is the transpose graph with all arcs reversed.

    Mathematical basis
    ------------------
    Reversing every arc induces a one-to-one correspondence between u->v paths
    in G and v->u paths in the transpose, preserving total cost.

    Test strategy
    -------------
    A strongly connected digraph is generated, reversed lazily with
    `G.reverse(copy=False)`, and the corresponding distances are compared.

    Assumptions / Preconditions
    ---------------------------
    The graph is generated as a directed graph with positive weights and is
    made strongly connected by construction, so both the original and the
    transposed distance queries are meaningful.

    Why this matters
    ----------------
    A directed shortest-path routine should behave consistently under graph
    transposition when source and target are swapped. A failure would indicate
    a bug in reverse-graph handling, adjacency traversal, or directed distance
    computation.
    """
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]
    assume(nx.has_path(G, u, v))
    G_rev = G.reverse(copy=False)
    assume(nx.has_path(G_rev, v, u))
    assert nx.dijkstra_path_length(G, u, v) == \
           nx.dijkstra_path_length(G_rev, v, u)


def _make_minimal_gbc_bug_graph():
    """Construct the smallest graph that reproduces the centrality bug."""
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (0, 2)])
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G


@pytest.mark.xfail(
    strict=True,
    reason=(
        "This test reproduces a bug in NetworkX 3.6.1: "
        "group_betweenness_centrality returns -1.0 on the 3-node digraph "
        "{(0,1),(0,2)} with C=[0]."
    ),
)
@_PBT_SETTINGS
@example(G=_make_minimal_gbc_bug_graph())
@given(_int_weighted_graph(directed=True, ensure_connected=False,
                           min_nodes=3, max_nodes=8))
def test_group_betweenness_centrality_non_negative(G):
    """
    Property (boundary invariant on group betweenness centrality)
    -------------------------------------------------------------
    Group betweenness centrality is a sum of non-negative fractions of path
    counts and must therefore never be negative.

    Mathematical basis
    ------------------
    The numerator and denominator in the definition are path counts, so each
    contribution is non-negative whenever defined. A negative total is not just
    surprising; it is mathematically impossible.

    Test strategy
    -------------
    Directed graphs that are not required to be strongly connected are
    generated, since unreachable pairs are precisely the fragile case. For
    C = {first node}, non-negativity is asserted. The minimal bug-triggering
    example is pinned with `@example` so the behaviour remains reproducible on
    every run.

    Assumptions / Preconditions
    ---------------------------
    The test assumes only that the graph has at least three nodes. Strong
    connectivity is deliberately not required, because the bug arises in the
    presence of unreachable node pairs.

    Discovered bug
    --------------
    On the digraph with edges {(0,1), (0,2)} and C=[0], NetworkX 3.6.1 returns
    -1.0, which violates the definition of the metric. This test is marked
    `xfail(strict=True)` so that an upstream fix becomes immediately visible as
    an XPASS rather than quietly passing by unnoticed.

    Why this matters
    ----------------
    This test checks a definitional invariant of the metric itself. A failure
    indicates not just an unexpected value, but a mathematically impossible
    output, which is strong evidence of a real bug in the centrality
    implementation.
    """
    nodes = list(G.nodes())
    assume(len(nodes) >= 3)
    C = [nodes[0]]
    val = nx.group_betweenness_centrality(G, C)
    assert val >= -1e-9, f"Negative group betweenness {val} for C={C}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
