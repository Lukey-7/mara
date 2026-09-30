---
title: Raft consensus
tags: [consensus, replication, raft]
published_date: 2026-09-01
---

# Raft consensus

Raft is a consensus algorithm for managing a replicated log. It was designed to be easier to
understand than Paxos while providing the same safety guarantees. A Raft cluster has one
leader at a time; all writes go through the leader.

## Roles and terms

Every server is a follower, candidate or leader. Time is divided into terms, numbered
monotonically. Each term begins with an election; at most one leader wins a term. Terms act
as a logical clock: a server that sees a higher term than its own immediately reverts to
follower and adopts the new term.

## Leader election

Followers expect periodic heartbeats from the leader. If a follower hears nothing for an
election timeout (typically 150–300 ms, randomised per server), it becomes a candidate,
increments its term, votes for itself and sends RequestVote to the others. A candidate that
receives votes from a majority becomes leader. Randomised timeouts make split votes rare:
usually one candidate times out first and wins before others start.

A server grants its vote only if the candidate's log is at least as up to date as its own
(compared by last log term, then last log index). This restriction is what guarantees the
new leader already contains every committed entry.

## Log replication

The leader appends a client command to its log and sends AppendEntries to followers. An
entry is committed once it is stored on a majority of servers; the leader then applies it to
its state machine and tells followers the new commit index in later heartbeats. AppendEntries
carries the index and term of the entry preceding the new ones; a follower rejects the call if
its log does not match at that position, and the leader backs up one entry at a time until the
logs agree. This consistency check makes logs identical up to the commit point.

## Safety properties

Raft guarantees Election Safety (one leader per term), Leader Append-Only (a leader never
overwrites its log), Log Matching (same index and term implies identical logs up to that
index), Leader Completeness (a committed entry appears in every future leader's log) and
State Machine Safety (no two servers apply different commands at the same index). A subtle
rule: a leader only commits entries from its own term by counting replicas; earlier-term
entries are committed indirectly when a later entry is committed.

## Membership changes and snapshots

Configuration changes use a joint consensus phase where both old and new majorities must
agree, preventing two disjoint majorities during the transition. Snapshots let servers discard
the log prefix already applied to the state machine; a leader sends InstallSnapshot to a
follower that is too far behind.
