# Hypothesis Testing for NetworkX

This repository contains a property-based test suite for selected NetworkX graph algorithms

## Submission Details

- Team Member Name: `Abijna Rao`
- SR No: `13-19-01-19-52-24-1-24725`
- Team Members: `1`

## Algorithms Covered

The test suite in `networkx_properties_tests.py` covers:

- Dijkstra shortest paths
- Bellman-Ford shortest paths
- Bidirectional Dijkstra
- Johnson's all-pairs shortest paths
- Minimum spanning tree
- Group betweenness centrality

## Test Design

The suite uses Hypothesis to generate weighted graphs and verify algorithmic properties instead of checking only a few fixed examples. The selected tests include:

- postcondition checks for Dijkstra path validity
- metamorphic testing through positive weight scaling
- cross-validation between Dijkstra and Bellman-Ford
- optimality and idempotence checks for minimum spanning trees
- an advanced predecessor-tree property for Dijkstra
- cross-validation for bidirectional Dijkstra and Johnson's algorithm
- a transposition property for directed shortest paths
- an `xfail` test documenting a real bug in `group_betweenness_centrality`

## Environment

The suite was written against:

- Python 3
- NetworkX 3.6.1
- Hypothesis 6.x
- pytest

## How to Run

Recommended command:

```bash
pytest -v networkx_properties_tests.py
```

Alternative command:

```bash
python3 networkx_properties_tests.py
```

## Expected Outcome

The suite currently contains 10 tests:

- 9 tests are expected to pass
- 1 test is marked `xfail` because it reproduces a bug in NetworkX 3.6.1 group betweenness centrality

## Current Run Output

```bash
pytest -v networkx_properties_tests.py
```

```text
test session starts
platform darwin -- Python 3.13.5, pytest-9.0.3, pluggy-1.5.0
plugins: hypothesis-6.152.1
collected 10 items

networkx_properties_tests.py::test_dijkstra_path_is_valid_walk_with_matching_length PASSED
networkx_properties_tests.py::test_dijkstra_positive_weight_scaling PASSED
networkx_properties_tests.py::test_dijkstra_equals_bellman_ford_on_nonneg_weights PASSED
networkx_properties_tests.py::test_mst_weight_le_any_spanning_tree PASSED
networkx_properties_tests.py::test_mst_idempotent PASSED
networkx_properties_tests.py::test_dijkstra_predecessor_tree PASSED
networkx_properties_tests.py::test_bidirectional_dijkstra_matches_dijkstra PASSED
networkx_properties_tests.py::test_johnson_matches_all_pairs_dijkstra PASSED
networkx_properties_tests.py::test_dijkstra_transposition_isomorphism PASSED
networkx_properties_tests.py::test_group_betweenness_centrality_non_negative XFAIL

9 passed, 1 xfailed in 1.23s
```
