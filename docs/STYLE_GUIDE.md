# Writing style guide: Sholto Douglas blog

Source: <https://sholtodouglas.github.io/>, fetched 2026-08-23. Corpus = the four full technical posts
(the fifth, `/linearised_state_reps/`, is a 169-word stub pointing at an older blog and is excluded from
the statistics):

| Post | Date | Words | Body paras (>=25w) | Mean words/para | Mean words/sentence | Figures | Inline `$math$` |
|---|---|---|---|---|---|---|---|
| [Does Hierarchial Reinforcement Learning work yet?](https://sholtodouglas.github.io/DoesHierarchialRLWorkYet/) | 2020-06-03 | 3688 | 42 | 74.4 | 25.0 | 16 | 17 |
| [Playing with Energy Models](https://sholtodouglas.github.io/PlayingwithEnergy/) | 2020-06-11 | 1930 | 21 | 73.4 | 24.9 | 8 | 23 |
| [Transfer learning from play and language - laying down the infrastructure](https://sholtodouglas.github.io/LearningFromPlayAndLanguage/) | 2020-12-30 | 2006 | 18 | 83.6 | 23.5 | 2 | 24 |
| [Transfer Learning from Play and Language - Nailing the Baseline](https://sholtodouglas.github.io/Learning-from-Play/) | 2021-03-02 | 2644 | 25 | 69.6 | 25.6 | 14 | 10 |

Local cached copies of the fetched HTML and stripped text used to derive every number below:
`/private/tmp/claude-501/-Users-stevenyang/3952dc21-32c7-4dd5-adb6-1c429cfe1c9c/scratchpad/*.txt`.

---

## 1. Typical post structure and section ordering

The template is stable across all four posts:

1. **Title** — a plain noun phrase or a blunt question. Titles are the *only* place a question mark
   is welcome ("Does Hierarchial Reinforcement Learning work yet?").
2. **Byline line**: `Sholto Douglas · June 3, 2020`, then 2-4 lowercase topic tags (`play`, `language`,
   `imitation`, `latent`).
3. **Hero artifact, above everything.** A GIF or plot of the *result*, immediately, before a single
   sentence of prose. Then a caption that already discloses the caveat ("This video is slightly cherry
   picked - the average success rate on this sequence of tasks is ~11/13.").
4. **A one-line code link** — "Code found here." / "Run the code on colab here." — plus collaborator
   credit if any. This sits in the first 40 words, not at the bottom.
5. **Auto-generated table of contents** (2 levels deep, 8-20 entries). This is the *only* place bullets
   dominate.
6. **Introduction**: the motivating question in a block quote, why it matters, what prior work he is
   building on (2-3 named papers with one-paragraph summaries), and — critically — a paragraph that
   states the bottom line and the failures *up front*, before any evidence.
7. **Body**: 3-6 `<h1>`-level sections, each with 2-4 `<h2>`/`<h3>` subsections. One experimental
   question per subsection. Each subsection is: setup prose → figure/GIF → interpretation prose that
   includes what did *not* work.
8. **"Whats next?" / "Next steps"** — concrete, dated-feeling plans, including the boring engineering
   ("we refactored the code", "deep diving into TPU utilisation and trialling FP16").
9. **Acknowledgements** — one sentence naming individuals.
10. **Appendix** — the sweeps and ablations that did not earn a place in the argument, each as a figure
    plus a two-sentence caption. Two of four posts have one.

Section headers are short imperative or descriptive fragments: "Diagnosing Overregularisation",
"Choosing Negative Examples", "Subgoal Testing", "Benefits beyond Exploration". Note that
question-form *subheaders* do occur (7 across the corpus: "What took us so long?", "What kind of
Subgoal performs best?", "At what level of complexity do our models tap out?") — but each is a real
experimental question the section then answers with a number or a plot, never an SEO-shaped
"Why does this matter?" filler header. If you cannot answer your header with a figure, do not use a
question.

## 2. The opening move

Three moves, always in this order, always inside the first ~120 words:

- **Show, then tell.** The GIF/plot is the first element in the DOM. The reader sees the result before
  the claim.
- **State the question as a quote.** Both play posts open the introduction with the literal research
  question set as a block quote: *"Can we enable fast transfer learning to new scenes or behaviours by
  using language to structure a joint trajectory embedding space between robot specific data and a much
  larger, diverse set of human video?"*
- **Situate against a named tension, then pre-announce the verdict.** "Hierarchial Reinforcment Learning
  (HRL) carries unrealised promise." → two paragraphs on why it should work → "Despite this, it has
  largely failed to deliver on the hype." → an OpenAI quote conceding hierarchy was unnecessary → then
  the two papers that made him reopen the question.

He never opens with a definition, never with "In recent years...", never with a hook about how fast the
field is moving.

## 3. Paragraph length and rhythm

- Body paragraphs: **mean 70-84 words, median ~65-76, max ~150-220**. Roughly **3 sentences** of
  **24-26 words each**. Sentences are long, but they are long because they are *loaded with conditions*,
  not because they are ornamental.
- Rhythm is: long compound sentence carrying the mechanism → short flat sentence carrying the verdict.
  "Long story short - RPY orientation control worked far better. Simplicity wins." / "This totally
  fails to learn." / "Next issue."
- One-sentence paragraphs are used sparingly, for pivots ("So, lets take a look at the impact of HAC -
  and what it takes to make it work.") and for jokes.
- **Dashes:** he uses the *spaced hyphen* ` - ` as his standard aside marker, ~1 per 60-140 words
  (42 in a 2644-word post). He uses the true em dash `—` essentially never (1 instance in 10,268 words,
  inside a quotation). The aside always adds a **fact or a caveat**, never a rhetorical flourish:
  "the difference was negligible", "but only around ~5-10!", "which would initially appear better".
- Bullets appear only for (a) the TOC, (b) enumerating the 2-3 mechanisms of an algorithm, (c) listing
  the experimental questions a section will answer. Never for prose that could be a paragraph. Across
  10k words there are roughly five genuine bullet lists.

## 4. First person and hedging

- **Person tracks authorship, exactly.** Solo posts are `I`/`my` (28 first-person-singular tokens in the
  HRL post, 21 in the energy post). Collaborative posts are `we`/`our` (49 in *Nailing the Baseline*,
  44 in *Laying down the infrastructure*, with `I` reserved for a personal anecdote: "I once heard that
  it takes abstract artists years..."). There is no passive-voice "it was observed that".
- **Hedges are epistemic and specific, and they name the alternative hypothesis.** Frequency is low —
  roughly 10-15 hedge tokens per post. The dominant forms are `likely`, `may`, `might`, `I suspect`,
  `we think`, `it is clear that X, but Y may need Z`. Example: "It is clear that the encoder has
  sufficient capacity to memorise (it's loss term tracks with the model which learns from states), but
  the planner may need increased capacity to be able to plan over the space, $\beta$ may need to be
  increased to bring the spaces together - or it may need to be trained for longer."
- **Affect words carry information, not enthusiasm.** `Curiously`, `Surprisingly`, `Interestingly`,
  `Unfortunately`, `frustratingly`, `This is truly bizzare` — each of these is attached to a specific
  measurement that violated a specific prior he had stated. They are *flags for the reader to slow
  down*, not filler.
- He asks the reader for help, in the first person, at the end: "I'd love any feedback you have on this
  piece. In particular, I'd love to know if the level of detail + referring to outside sources is
  appropriate, or if more explaination is necessary."
- Typos and British/idiosyncratic spellings survive ("Hierarchial", "bizzare", "explaination",
  "possiblity"). Do not imitate the typos, but do inherit the implication: the post is a lab notebook
  made public, not a press release.

## 5. Results, plots and tables

- **Figures carry the argument; prose carries the interpretation.** 8-16 figures per post, i.e. roughly
  one visual per 170-230 words. GIFs of rollouts are the primary evidence for behaviour; training-curve
  panels and latent-space scatter plots are the primary evidence for mechanism.
- **Captions are full sentences that state the surprise**, not labels. "Curiously, the same
  regularisation term was not sufficient to converge the planner and encoder latent spaces from
  pixels.." / "The sweet spot for regularisation does not directly follow from reconstruction loss."
  / "Plots like these are used to diagnose overregularisation during the training process."
- **Tables are rare and grudging** — one across the corpus, and it is raw
  (`-193.80689655172415  0.0896551724137931`). The lesson: prefer a plot; if you must table, table few
  rows. (Do round. He didn't; you should.)
- **Every metric arrives with its measurement protocol and its known failure mode in the same
  paragraph.** He defines success ("all elements of the state within 5cm and 30 degrees"), states the
  budget ("the robot only receives 4s"), states what the metric misses ("fails to account for behaviour
  which is mostly correct"), gives the counterfactual number ("Adjusting for this would bring the
  success rate of these tasks to ~85%"), and then explicitly downgrades the claim to what it can
  support ("we think it suffices as a relative comparison").
- **Numbers are quoted loosely on purpose** when precision is not the point: `~11/13`, `>90%`, `2-3x as
  fast`, `10x less expert demonstrations (2000 timesteps ... vs 20000 timesteps)`. He does not
  manufacture significant digits — but he always gives the *scale* of an effect, never just its
  direction.
- **Control experiments get their own sentence.** See excerpt 2 below: he anticipates the confound,
  measures it, and reports that it was negligible.

## 6. Negative results and uncertainty

This is the load-bearing part of the style. Rules he follows without exception:

1. **Nulls go in the introduction, not the conclusion.** "Unfortunately, I failed to succeed at more
   complex environments ... but hey, RL is hard."
2. **Name the most likely cause of your own failure, and say what would test it.** "The most likely
   issue is that the off-policy HRL framework I am using is too unstable compared to the on-policy
   algorithm used in RPL. With the recent release of the architecture specifics and simulation
   environment of RPL I plan to revisit this."
3. **Report the diagnostic you hoped for and did not get.** "This is frustratingly qualitative. We were
   hoping that the encoder and planner reconstruction losses would converge to a lower final value ...
   but this wasn't the case."
4. **Disclose bugs that produced results.** "Surprisingly, subgoal testing with the achieved goal
   instead of the original goal performs best. This is truly bizzare, and actually arose from a bug in
   my code where I was still subsituting the action when subgoal testing."
5. **Confirm other people's negative results explicitly.** "My results here match theirs - there is no
   significant difference when the models are trained with the same transitions."
6. **Bound the claim, out loud.** "This is by no means the final word on HRL, but how easy and effective
   reimplementing the work is gives a good idea of how robust the area is."
7. **Say what you will stop doing.** The energy-model post ends by abandoning the direction: "EBMs have
   a lot of promise and have achieved extremely powerful results - but for my RL interests, I'm going to
   continue to experiment with other ideas."
8. **Post updates when corrected.** The energy post has an `Update` section crediting Yilun Du with a
   pointer to a better formulation, and says which work he would build on next.
9. **Flag known weaknesses in rigour rather than hiding them.** "We could have done this more rigorously
   - but we wanted to keep progressing."

## 7. Equations vs prose

- **Sparse and functional.** 10-24 inline `$...$` spans per post; typically **zero to two display
  equations**, and those only when the equation *is* the contribution being discussed
  (`$\hat{y} = \min_{y} F(x,y)$`, the margin-ranking loss, the RL objective
  `$E_{\pi}(\sum_{t=0}^T \gamma^t R(s_t,a_t))$`).
- Notation is introduced in-line inside a sentence, immediately glossed in words:
  "the standard formulation of RL involves an environment with transition function $P(s_{t+1}|s_t,a_t)$,
  where $s_t$ and $a_t$ are the states and actions at timestep t".
- Symbols are reused as prose nouns thereafter ("Too high, and the latent space collapses").
- Ratio is roughly **95% prose / 5% math**. Nothing is derived. If a mechanism can be said in a
  sentence, it is said in a sentence.

## 8. Length and register

- **Target 1,900-3,700 words; ~2,500 is the centre of mass.** Structural posts (infrastructure,
  negative-result surveys) sit at the top of the range; single-idea experiment posts at the bottom.
- **Assumed reader:** a working ML researcher or strong grad student in the *adjacent* subfield. He
  assumes you know what a VAE, a KL term, an LSTM, off-policy RL and a replay buffer are, and does not
  define them. He does *not* assume you have read the specific papers under discussion — those get a
  one-paragraph summary each, plus a link ("The Gradient published an excellent overview.").
- Register is **peer-to-peer lab conversation**: informal, self-deprecating, technically unhedged about
  facts and heavily hedged about causes. Occasional jokes and asides are in-bounds ("isn't kinetic
  punishment the fastest way to a robot's heart?", "Not unless you're trying to imitation learn off
  Houdini"), but they are always adjacent to a real technical point, never a substitute for one.
- Links are frequent and are load-bearing: papers, code repos, Colab notebooks, other people's blog
  posts. Named individuals are credited by name in prose.

## 9. Characteristic phrasings (use these)

- "Once again - the answer wasn't in [list of clever fixes] ... it lay in more fundamental fixes:"
- "To verify that this effect was due to X, and not Y - we [measurement], but the difference was
  negligible."
- "To quantify this, we defined ..."
- "This is frustratingly qualitative."
- "This is by no means the final word on X."
- "It would be interesting to test whether ..."
- "The most likely issue is ..."
- "What this means is that ..."
- "Curiously, ..." / "Surprisingly, ..." / "Unfortunately, ..."
- "I'm extremely curious what error I'm making."
- "but hey, RL is hard"
- "Long story short - ... Simplicity wins."

## 10. Anti-patterns — never do these

**Vocabulary that appears zero times in 10,268 words of his writing:** *delve, it is important to note,
in conclusion, moreover, notably, arguably, key takeaway, dive into, unlock, harness, paradigm,
landscape, testament, game-changer, seamless, cutting-edge, comprehensive.* (Total hits across the whole
corpus: `furthermore` 2, `robust` 6 — and `robust` is always used literally, of a model or a research
area.) Treat that list as a hard blocklist.

Structural anti-patterns:

- **Bullet-point soup.** Do not convert findings into a deck of bullets. His findings are paragraphs; his
  bullets are enumerations of mechanisms. If a bullet has a bolded lead-in phrase followed by a colon
  and a sentence, you have written LLM slop, not this.
- **Breathless hype.** No "This is huge", "remarkable", "striking", "a fundamental shift". The strongest
  intensifiers in the corpus are "dramatically improves", "significantly more expressive", and
  "truly bizzare" — each attached to a measured quantity or a specific anomaly.
- **Em-dash-heavy hedging.** He uses ` - ` for factual asides, not `—` for rhetorical balancing.
  Never write "not X — but rather Y" as a stylistic tic. If an aside does not add a measurement, a
  caveat, or a named alternative, delete it.
- **Question-shaped section headers that the section does not answer.** "Why does this matter?",
  "What's next for the field?", "So what?" are banned. A question header is allowed only if the section
  closes it with a plot or a number.
- **Symmetric three-part conclusions.** No "In conclusion, we have shown three things." His conclusions
  are asymmetric and downbeat: what held, what didn't, what he is doing next.
- **Hedging that hides the result.** Never "results were mixed" or "further work is needed" without the
  specific number and the specific next experiment.
- **Rhetorical questions as transitions.** ("But what if we could go further?") The only questions in
  his prose are ones he then measures.
- **Defining basics the audience already knows**, or conversely dumping notation without a gloss.
- **Burying the negative result.** Do not save the failure for a final "Limitations" section.
- **Headline claims without the protocol.** Never report a success rate without saying what counts as
  success and how long the model got.
- **Perfect grammar-of-a-press-release voice.** Contractions, "ok", "a bit of fun", "hey" all belong.

## 11. Five verbatim calibration excerpts

> **1 — opening a post by conceding the failures (HRL, Introduction)**
>
> "This blog post uses two test environments. In the first a pointmass must push a block to a target
> position . This is an ideal testing environment because it is fast to train but contains basic versions
> of the difficulties facing robotic manipulation tasks (namely, that working out how to even manipulate
> the block requires significant exploration of the environment). Unfortunately, I failed to succeed at
> more complex environments, such as the same task but with multiple blocks and my robotic manipulation
> environment, but hey, RL is hard. The most likely issue is that the off-policy HRL framework I am using
> is too unstable compared to the on-policy algorithm used in RPL. With the recent release of the
> architecture specifics and simulation environment of RPL I plan to revisit this."

> **2 — pre-empting a confound in one sentence (Nailing the Baseline)**
>
> "To verify that this effect was due to the the behaviour demonstrated, and not that a multi-interaction
> dataset provides more timesteps of interaction with the environment - we counted the proportion of
> timesteps where an environment variable was different to the previous state (i.e, arm interacting not
> transitioning), but the difference was negligible."

> **3 — reporting the null you did not want (Nailing the Baseline)**
>
> "This is frustratingly qualitative. We were hoping that the encoder and planner reconstruction losses
> would converge to a lower final value for the well regularised models, even if the planner improved
> more slowly than for over regularised models - but this wasn't the case."

> **4 — closing a post by abandoning the direction (Playing with Energy Models)**
>
> "I found it interesting to do a set of quick experiments with energy functions for generating states to
> plan with, but my results were not very compelling. Firstly, the generated states didn't include the
> direct straight line, optimal path - even in a very simple problem. Secondly, even spiral generation was
> flawed. EBMs have a lot of promise and have achieved extremely powerful results - but for my RL
> interests, I'm going to continue to experiment with other ideas rather than pursuing the path of trying
> to get this framework to work well."

> **5 — voice: the joke that carries the engineering decision (Laying down the infrastructure)**
>
> "Hold up! We had been playing with too many satellites. Does a robot's end effector really need to go
> beyond $\pm \pi$ in any axis of rotation? Will it ever encounter a discontinuity? Not unless you're
> trying to imitation learn off Houdini. We already had a link positioned at the tips of the gripper for
> inverse kinematics purposes - all you need to do is rotate that so that it is at 0,0,0 RPY in the arm's
> default pose."

---

## 12. Writing for Pratyusha Sharma specifically

The eventual post is aimed at Pratyusha Sharma — incoming assistant professor at NYU, senior research
scientist at Microsoft Research, previously MIT CSAIL. Relevant work:

- **"LoRA vs Full Fine-tuning: An Illusion of Equivalence"** (Shuttleworth, Sharma, Andreas, Torralba;
  arXiv [2410.21228](https://arxiv.org/abs/2410.21228), NeurIPS 2025). LoRA-trained weight matrices
  acquire new high-ranking singular vectors — *intruder dimensions* — that full fine-tuning does not
  produce; intruder dimensions are shown *causally* to drive forgetting by intervening on their singular
  values post-hoc.
- **"The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning"**
  (arXiv [2607.23711](https://arxiv.org/abs/2607.23711)) — the follow-on scaling story.
- **OP-Mix**, "Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time"
  (Hu, Gandhi, Cho, Linzen, Sharma; arXiv [2605.15220](https://arxiv.org/abs/2605.15220)) — data mixtures
  simulated cheaply by interpolating between low-rank adapters; matches retraining and on-policy
  distillation for continual learning at 66% / 95% less compute.

*(These citations were retrieved via web search on 2026-08-23; verify each against the arXiv listing
before the post ships.)*

What a reader with that profile needs, on top of the Sholto template:

1. **Preregistration clarity.** State H1/H2/H3 and the decision rule *before* the first plot, and say
   plainly which hypotheses were fixed in the spec versus formed after seeing data. Cite the spec file
   and the pinned commit. She works on causal interventions; she will read post-hoc hypotheses as
   post-hoc unless you mark them.
2. **Honest nulls, promoted.** A confirmed null on the H3 near-vs-far interference experiment is a
   result and should be reported at the same visual weight as a positive H1 geometry effect — in the
   introduction, not an appendix. Say explicitly what would have counted as a positive.
3. **Effect sizes with intervals, not p-values alone.** Three seeds means seed-level scatter, not error
   bars implying a large n. Report per-seed values, the across-seed spread, and a standardised effect
   size where one is meaningful. State the power you actually have — with three seeds, say what
   magnitude you could and could not have detected.
4. **Explicit links to the forgetting / spectral agenda.** Draw the connection in her vocabulary, and be
   careful not to overclaim it. Concretely: (a) NextLat's latent-transition head is a *representational*
   constraint, whereas intruder dimensions are a *parameter-space* one — say whether your geometry
   metrics are, or are not, a representational analogue of her spectral diagnostic; (b) if you measure
   history separation and later interference, frame interference in the forgetting language she uses
   (localised vs distributed) and say whether you can localise it; (c) the honest position is likely
   "this is an analogy worth testing, and here is the specific measurement that would settle it" —
   which is exactly the Sholto register anyway.
5. **Give her the reproduction path in the first 40 words**, as he does: pinned upstream commit
   (`3770be6009cea2b3c455a9ce7f2ca88b504bb955`), configs, seeds, GPU type, wall-clock, artifact hashes.
6. **No neuroscience overreach.** The project spec forbids turning the memory/interference framing into
   biological claims; a research reader will discount the whole post at the first unearned analogy.
7. **Name the thing you would do next with more compute**, and what you would want from her line of work
   to do it — a real, specific ask, not "future work".
