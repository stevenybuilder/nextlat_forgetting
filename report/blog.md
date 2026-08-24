# Does NextLat separate the histories that matter?

*Draft in progress. Every `[pending]` below is a number that has not been measured yet - the
document is generated from `results/live_numbers.json`, so a claim cannot quietly outrun its
artifact.*

![Final-state PSI for GPT and NextLat across three seeds. Higher means the model pushes apart histories whose futures differ, relative to equally-perturbed histories whose futures agree.](results/figures/fig2_psi.png)

Code and manifests: `github.com/<pending>/nextlat-lurestar`. Everything here runs on one GPU.

## The question

> Does NextLat selectively separate similar histories when a matched change alters the correct
> future, keep equally changed but future-equivalent histories closer, and does weak separation
> predict later interference?

[NextLat](https://arxiv.org/html/2511.05963) adds a latent transition model to next-token
training: a network `p_psi` takes the hidden state `h_t` and the next token `x_{t+1}` and predicts
the next hidden state, and the transformer is trained so that its own states are what the
transition model predicts. Under the paper's idealised optimisation assumptions the hidden state
becomes sufficient for predicting the future - a belief state. Empirically it solves Path-Star,
tracks state better, generalises to longer sequences, and gets a speculative-decoding speedup.

What the paper does not do is say what the resulting representation space looks like. That is not
my framing, it is theirs, from section 6:

> On the analysis side, we do not study the structure of the learned representations under
> NextLat, leaving open questions about how the method shapes latent spaces.

So the interesting question is not whether NextLat compresses - the paper already shows an
effective latent rank of 52.7 against GPT's 160.1 on Manhattan, and reproducing that would be
reproducing their result. The question is *what gets compressed together and what gets pushed
apart*, and whether that geometry has downstream consequences.

## The bottom line, before the evidence

{{live:bottom_line}}

## Why a matched lure is the whole experiment

The failure mode of any representation-geometry claim is that you measure a nuisance. If you
compare a base graph against a perturbed graph and find the states moved, you have learned that
editing the input moves the state - which is not a finding. The comparison has to hold the size
of the edit fixed and vary only whether the edit *changes the correct future*.

Path-Star gives a clean way to do this. A `G(5,5)` graph has a source and five disjoint arms of
four edges, serialised as a shuffled edge list plus a source and goal, and the model has to emit
the path. Take two arms and swap their suffixes at equal depth: the edges `u->a` and `v->b`
become `u->b` and `v->a`. Exactly two endpoint tokens change. The node multiset is preserved, the
degree sequence is preserved, the prompt and answer lengths are preserved.

Now the whole design turns on which arms you pick. Swap between two distractor arms and the
correct path is untouched - call it **near-safe**. Swap between the goal arm and a distractor and
the goal token stays where it is, but the correct first branch changes - **near-critical**. Two
edits of identical size, one of which matters for the future and one of which does not.

{{live:stimulus_matching_summary}}

## H1 - future-sensitive separation

{{live:h1_summary}}

![Safe versus critical behaviour by condition.](results/figures/fig1_behaviour.png)

## H2 - does the geometry predict planning?

{{live:h2_summary}}

![Representation distance against critical-branch margin, held out under two-fold cross-fitting.](results/figures/fig3_distance_margin.png)

## H3 - does weak separation predict interference?

This is the part that connects to work on forgetting. From each frozen base checkpoint I branch
twice, adapting on lures that are *near* the trained items and on lures that are *far*, matched
on adaptation examples, update count, initial loss quantiles, target-path distribution, item
order, optimiser and scheduler state, learning rate, and batch size. Both branches are
full-parameter next-token-only adaptation, with NextLat's auxiliary losses switched off so that
what is being tested is the base representation rather than ongoing regularisation.

{{live:h3_summary}}

![Near versus far interference and acquisition.](results/figures/fig4_interference.png)

## The HMM, where the belief state is known exactly

Path-Star cannot tell you what the sufficient predictive state actually is. A small HMM can. With
a preregistered 4-state, 4-observation chain with overlapping emissions, the normalised forward
algorithm gives the exact posterior over hidden states for every prefix, so predictive equivalence
is a fact about the generative process rather than an inference from the model.

{{live:hmm_summary}}

![Hidden-state distance against exact belief JS divergence.](results/figures/fig5_hmm_js.png)

![Predictive-equivalence collapse and posterior decoding.](results/figures/fig6_hmm_decoding.png)

A caveat that has to stay attached to every one of these numbers: a sufficient predictive state is
not unique. An invertible transformation preserves every bit of predictive information while
changing raw Euclidean geometry entirely. That is why the tests here are about predictive
equivalence, relative divergence, decodability and future-distribution prediction, and not about
whether the states line up with the belief simplex.

## What did not work

{{live:failures}}

## Whats next?

{{live:next_steps}}

## Notes on the engineering

{{live:engineering_notes}}

## Acknowledgements

{{live:acknowledgements}}

## Appendix - preregistration and what would have falsified this

{{live:preregistration}}
