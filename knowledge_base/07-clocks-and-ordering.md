---
title: Clocks, ordering and vector clocks
tags: [ordering, clocks, theory]
published_date: 2026-09-04
---

# Clocks, ordering and vector clocks

## Why physical clocks are not enough

Machine clocks drift and NTP corrections can jump backwards. Two nodes cannot agree on which
of two events happened first by comparing timestamps unless the clock error bound is known.
Last-writer-wins by wall clock silently drops writes when clocks disagree.

## Lamport timestamps

Each node keeps a counter; it increments the counter for every local event and, on receiving
a message, sets its counter to max(local, received) + 1. If event a caused event b, then
L(a) < L(b). The converse does not hold: two events with L(a) < L(b) may be concurrent.
Lamport clocks give a total order consistent with causality, useful for tie-breaking, but
cannot detect concurrency.

## Vector clocks

Each node keeps a vector with one counter per node. On a local event it increments its own
entry; on send it attaches the vector; on receive it takes the element-wise maximum and then
increments its own entry. V(a) < V(b) (every entry ≤, at least one <) if and only if a
happened before b; if neither dominates, the events are concurrent. Dynamo used vector clocks
to detect conflicting writes and hand siblings back to the application. The vector grows with
the number of writers, so dotted version vectors and pruning are needed in practice.

## Hybrid logical clocks

HLC combines a physical timestamp with a logical counter: it stays close to wall-clock time
(so timestamps are meaningful to humans and can be compared across systems) while preserving
causality like a Lamport clock. CockroachDB uses HLC for transaction timestamps.

## TrueTime

Spanner's TrueTime API returns an interval [earliest, latest] guaranteed to contain the true
time, using GPS and atomic clocks in each data centre. A transaction waits out the uncertainty
(commit wait, a few milliseconds) before releasing locks, which makes timestamps externally
consistent: if T1 commits before T2 starts, T1's timestamp is smaller.
