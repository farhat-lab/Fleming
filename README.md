![icon](figures/icon.png)

-----------------

# Fleming: An AI Agent for Antibiotic Discovery In Mycobacterium Tuberculosis

## What is Fleming?
Fleming is an agentic system for designing, prioritizing, and optimizing novel small-molecule antibiotics against Mycobacterium tuberculosis (Mtb). It combines tool-using LLM orchestration with state-of-the-art molecular ML models, literature-aware reasoning, and generative chemistry to support both:

* Co-pilot mode: interactive analysis/design of a small number of molecules in real time

* Batch mode: high-throughput screening, generation, ranking, and optimization of large libraries
![fleming](figures/fleming.png)

# Overview 
Fleming is centered around a manager (“medicinal chemist”) agent that routes user requests to specialized sub-agents and combines outputs into a final decision.

### TB Growth Inhibition Agent

Predicts whole-cell Mtb growth inhibition using an evidential DMPNN (eDMPNN) trained on 114,933 compounds/fragments with in vitro growth readouts.

Provides uncertainty-aware predictions (useful for out-of-distribution molecules).

Performs literature retrieval and synthesis to modify predictions based on biological priors. 

Produces model explanations (e.g., substructure attribution) and LLM-based plausibility checks.

### ADMET Agent
Predicts absorption/distribution/metabolism/excretion/toxicity using ADMET.AI models across many endpoints.

Performs literature-grounded reasoning (literature retrieval + synthesis) to improve robustness, especially when model confidence is low or chemistry is novel.
### Molecular Generation Agent
Designs new candidate inhibitors using:

Conditional equivariant diffusion

SyntheMol / MCTS-style fragment generation

Filters generations for novelty and synthetic feasibility (SAScore/SCScore).

### Molecular Optimization Agent
Molecular Optimization Agent

Improves ADMET properties via local chemical neighbor search / modification while preserving predicted inhibition.

Supports multi-objective optimization (e.g., reduce hepatotoxicity + avoid hERG + keep inhibition high).


## Repo Structure

* `fleming` contains the full AI agent 
* `evaluation` contains evaluation scripts

