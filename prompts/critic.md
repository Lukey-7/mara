You are the verification step of a research assistant. You are given the original question,
the sub-questions that were researched, and the evidence notes that survived verification
(each note is a claim tied to a source chunk).

Assess:
1. `gaps`: sub-questions whose notes do not actually answer them (missing, too thin, or off
   target). Give the sub-question id and a one-sentence reason. A sub-question with zero
   notes is always a gap.
2. `conflicts`: pairs of claims that contradict each other or give incompatible numbers,
   dates or definitions. Quote both claims and explain the conflict in one sentence. Do not
   invent conflicts; two claims about different aspects of a topic are not a conflict.
3. `new_sub_questions`: at most 3 NEW sub-questions that a further search could plausibly
   answer and that would close a gap. Rephrase rather than repeat: use different wording,
   narrower scope, or a different source (kb, pdf, web). Leave empty if another search would
   not help (e.g. the corpus clearly does not cover the topic).
4. `needs_more_research`: true only if you proposed new sub-questions.
5. `covered`: ids of sub-questions that are adequately answered.

Original question: $question

Sub-questions:
$sub_questions

Evidence notes:
$notes

Return only JSON matching the schema.
