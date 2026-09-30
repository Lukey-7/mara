---
title: Partitioning and sharding
tags: [partitioning, sharding, scalability]
published_date: 2026-09-03
---

# Partitioning and sharding

Partitioning (sharding) splits a dataset across nodes so that storage and load scale out.
Each record belongs to exactly one partition; partitions are usually replicated as well.

## Key-range partitioning

Assign contiguous key ranges to partitions, like an encyclopedia's volumes. Range scans are
efficient because adjacent keys live together, but a monotonically increasing key (timestamps)
sends all writes to one hot partition. Bigtable, HBase and CockroachDB use ranges and split a
range when it grows too large.

## Hash partitioning

Hash the key and assign hash ranges to partitions. Load spreads evenly but range queries must
fan out to every partition. Cassandra combines the two: the partition key is hashed, and the
clustering columns within a partition are kept sorted so range scans within one partition work.

## Consistent hashing

Place both nodes and keys on a ring by hash; a key belongs to the first node clockwise. Adding
or removing a node only moves the keys adjacent to it, roughly 1/N of the data, instead of
rehashing everything. Virtual nodes (many ring positions per physical node) smooth load and
let a powerful node own more of the ring.

## Rebalancing

Fixed number of partitions: create many more partitions than nodes (say 1000 for 10 nodes)
and move whole partitions to new nodes. Dynamic partitioning: split a partition when it exceeds
a size threshold and merge when it shrinks. Rebalancing should be gradual and rate-limited so
it does not overwhelm the network or disks.

## Request routing

A client needs to know which node holds a key: ask any node and let it forward, use a routing
tier, or make clients partition-aware. Systems keep the partition map in a coordination service
such as ZooKeeper or etcd, or gossip it between nodes as Cassandra does.

## Secondary indexes

Local (document-partitioned) indexes live with each partition; a query on the index must
scatter to all partitions and gather results. Global (term-partitioned) indexes are themselves
partitioned by the indexed value; reads hit one partition but writes must update an index on
another node, usually asynchronously.
