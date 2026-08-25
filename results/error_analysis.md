# Qualitative error analysis

Systems compared: transformer_warmup_xavier vs rnn_xavier, both beam-5, test set.
Selection rule pre-committed and run once (src/error_analysis_select.py), before reading
any translation text: highest min(per-sentence BLEU) for "both correct", lowest max(per-
sentence BLEU) for "both wrong", highest absolute per-sentence BLEU gap for "systems
disagree most", plus two indices sampled uniformly at random with a fixed seed. Five
examples, not six - the fifth (random) slot doubled as a second, independently useful
disagreement case, so no sixth was added.

## 1. Both correct (idx 89)

SRC (vi): Cám ơn các bạn .
Gloss: thank-you PLURAL-you .
REF: Thank you .
Transformer: Thank you .
RNN: Thank you .

A short, common phrase - both systems reproduce it exactly. Included as the required
positive control: the systems are not universally bad, and this rules out "neither system
ever works" as an alternative explanation for the BLEU gaps reported elsewhere.

## 2. Both wrong (idx 39)

SRC (vi): vì mặc dù đã bị bắt , nhưng cuối cùng học cũng được thả ra nhờ vào sức ép từ cộng
đồng quốc tế .
Gloss: because although already PASSIVE-arrest , but finally PRONOUN also PASSIVE-release
thanks-to pressure from community international .
REF: Even though they were caught , they were eventually released after heavy
international pressure .
Transformer: Because even though it was arrested , but at the end of science was released
by the international community .
RNN: Because even though it was arrested , but eventually it was released by the force of
the international community .

Two separate, identifiable error types, shared by both systems: (1) both keep "Because ...
but" as a paired connective, calquing the Vietnamese "vì ... nhưng" construction directly
into ungrammatical English - the reference drops the connective entirely instead. (2) the
source word "học" (most literally "study/learn") is very likely a transcription artifact
for "họ" (they/them, the subject the reference actually uses) - a source-data noise issue,
not a model failure - and the two systems handle the resulting bad input differently:
Transformer hallucinates an unrelated word ("science"), RNN drops the subject and produces
a subjectless clause instead. Worth naming as a source-data quality caveat, since the
"wrong" translation here is arguably an unanswerable question given the corrupted input.

## 3. Systems disagree most (idx 567)

SRC (vi): Vỗ tay Cám ơn
Gloss: clap-hands thank-you (a transcript stage direction "[Applause]" run directly into
the closing "Thank you", no punctuation between them in the source)
REF: Thank you .
Transformer: (empty)
RNN: Thank you .

The single most dramatic gap in the whole test set (0 vs 100 sentence-BLEU) is not a
subtle translation-quality difference - the Transformer produces an empty hypothesis
entirely, most likely because this unusual, unpunctuated stage-direction-plus-utterance
input falls far outside normal sentence shape and drives the decoder to emit end-of-
sequence immediately. RNN, on this same malformed input, still recovers a fluent, correct
short answer. A genuine robustness difference on out-of-distribution input, not evidence
that RNN out-translates the Transformer in general - the aggregate test BLEU numbers say
the opposite.

## 4. Random sample 1 (idx 228)

SRC (vi): Và anh ta nói rằng anh ta cần những cây súng này bởi vì những tổn thương mà anh
đã trải qua trong quá khứ khi là một đứa trẻ .
Gloss: and he say that he need CLASSIFIER gun this because CLASSIFIER trauma/injury that
he PAST go-through in past-time when be one child .
REF: And he said that he needed those guns because of the trauma he 'd experienced as a
young boy .
Transformer: And he said he needed these guns because the damage that he had gone through
the past when he was a child .
RNN: And he said that he needed these guns because he had experienced in the past as a
child .

Transformer keeps the object ("the damage") and is mostly complete, with an awkward
missing preposition ("because the damage" instead of "because of the damage"). RNN drops
the object entirely - "he had experienced" with nothing after it - leaving a transitive
verb without what it acted on, a real content-omission error rather than a grammar slip.

## 5. Random sample 2 (idx 51)

SRC (vi): Tôi là người Nam Triều Tiên hay Bắc Triều Tiên ?
Gloss: I am person South Korea or North Korea ?
REF: Am I South Korean or North Korean ?
Transformer: I am South Korea or North Korea ?
RNN: I am North Korea or North Korea ?

Transformer correctly keeps the South/North contrast (minor grammar: "South Korea" instead
of the adjective "South Korean", and no question inversion). RNN mistranslates the first
occurrence too, rendering both halves as "North Korea" - a genuine lexical confusion on a
minimal-pair directional term ("Nam" = south vs "Bắc" = north), not a fluency issue.
