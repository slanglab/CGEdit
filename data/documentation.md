## Dataset contents

### CGEdit/AAE.tsv
- 474 items
- Each item represents a training example in the form of an utterance
- For each item, there are binary variables representing the presence of 17 linguistic features, where 1 indicates the feature is present and 0 indicates the feature is not present. Example utterances for each linguistic feature can be found in Table 3 (Appendix A) of the paper. The following are mappings of column names to the linguistic features:
	1. zero-poss: Zero possessive -*'s*
	1. zero-copula: Zero copula
	1. double-tense: Double marked/overregularized
	1. be-construction: Habitual *be*
	1. resultant-done: Resultant *done*
	1. finna: *finna*
	1. come: *come*
	1. double-modal: Double modal
	1. multiple-neg: Negative concord
	1. neg-inversion: Negative axuiliary inversion
	1. n-inv-neg-concord: Non-inverted negative concord
	1. aint: Preverbal negator *ain't*
	1. zero-3sg-pres-s: Zero 3rd p sg present tense -*s*
	1. is-was-gen: *is/was*-generalization
	1. zero-pl-s: Zero plural-*s*
	1. double-object: Double-object construction
	1. wh-qu: *Wh*-question

### CGEdit-ManualGen/AAE.tsv
- 550 items
- Same information about items as CGEdit/AAE.tsv

### CGEdit/IndE.tsv
- 285 items
- Each item represents a training example in the form of an utterance
- For each item, there are binary variables representing the presence of 10 linguistic features, where 1 indicates the feature is present and 0 indicates the feature is not present. Example utterances for each linguistic feature can be found in Table 2 (Appendix A) of the paper. The following are mappings of column names to the linguistic features:
	1. foc\_self: Focus *itself*
	1. foc\_only: Focus *only*
	1. left\_dis: Left dislocation
	1. non\_init\_exis: Non-initial existential *there*
	1. obj\_front: Topicalized object (argument)
	1. inv\_tag: Invariant tag *no/na/isn't it*
	1. cop\_omis: Zero copula
	1. res\_obj\_pron: Resumptive object pronoun
	1. res\_sub\_pron: Resumptive subject pronoun
	1. top\_non\_arg\_con: Topicalized non-argument constituent 

### CGEdit-ManualGen/IndE.tsv
- 333 items
- Same information about items as CGEdit/IndE.tsv



