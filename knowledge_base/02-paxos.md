---
title: Paxos
tags: [consensus, paxos]
published_date: 2026-09-01
---

# Paxos

Paxos is a family of protocols for reaching agreement on a single value among unreliable
processes. Lamport published it in 1998; it underpins Google Chubby and many replicated
services. It tolerates f crash failures with 2f + 1 acceptors.

## Roles

Proposers propose values, acceptors vote, and learners find out which value was chosen. A
single process usually plays all three roles. Safety holds with any number of proposers;
liveness needs a distinguished proposer (leader) so that competing proposals do not livelock.

## The two phases

Phase 1 (prepare): a proposer picks a proposal number n greater than any it has used and sends
Prepare(n) to a majority of acceptors. An acceptor that has not promised a higher number
replies with a promise not to accept proposals below n, plus the highest-numbered proposal it
has already accepted, if any.

Phase 2 (accept): if the proposer hears from a majority, it sends Accept(n, v) where v is the
value of the highest-numbered proposal reported in the promises, or its own value if none was
reported. Acceptors accept unless they have since promised a higher number. A value is chosen
once a majority has accepted the same proposal number.

## Why it is safe

Any two majorities intersect. If a value v was chosen with number n, every later proposer's
prepare phase reaches at least one acceptor that accepted v, and the rule to adopt the
highest-numbered reported value forces the proposer to re-propose v. So once chosen, a value
can never be overturned.

## Multi-Paxos

Running the full protocol for every command costs two round trips. Multi-Paxos elects a
stable leader that runs phase 1 once for an unbounded sequence of log slots and then only
runs phase 2 per command: one round trip in the common case. This is essentially what Raft
does, with a more prescriptive log structure and election rule.

## Paxos versus Raft

Both tolerate minority failures and need a majority to make progress. Raft fixes the log
structure and only allows a server with an up-to-date log to become leader; Paxos allows any
proposer and reconciles log holes afterwards. Raft is generally considered easier to implement
correctly; Paxos variants (EPaxos, Flexible Paxos) offer more flexibility, such as separate
quorum sizes for elections and replication.
