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
