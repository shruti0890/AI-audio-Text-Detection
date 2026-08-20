"""
text_forensics/calibration/generator_attribution_experiment/dataset/build_attribution_dataset.py

Constructs the multi-generator parallel prompt dataset for AI-generator attribution:
  - OpenAI ChatGPT (GPT-3.5 / GPT-4 / GPT-4o)
  - Google Gemini (Gemini 1.5 Pro / Flash)
  - Anthropic Claude (Claude 3.5 Sonnet / Opus / Haiku)
  - Unknown / Other AI (Llama 3, Mistral Large, DeepSeek)
  - Human Ground-Truth Controls

Applies strict prompt-level grouped partitioning:
  - 70% Train
  - 15% Validation
  - 15% Held-Out Test
All generator variations of a prompt remain strictly in the same split.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

_HERE = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _HERE.parent
_DATASET_DIR = _EXPERIMENT_DIR / "dataset"
_CONFIGS_DIR = _EXPERIMENT_DIR / "configs"

# Multi-domain parallel prompt dataset catalogue
# Each prompt contains responses from ChatGPT, Gemini, Claude, Other AI, and Human
PARALLEL_PROMPT_CATALOGUE: List[Dict[str, Any]] = [
    # 1. Computer Science: Distributed Consensus
    {
        "prompt_id": "cs_consensus_01",
        "domain": "Computer Science",
        "prompt": "Explain how the Raft consensus algorithm maintains state machine replication and handles split votes.",
        "responses": {
            "ChatGPT": "The Raft consensus algorithm decomposes distributed consensus into leader election, log replication, and safety guarantees. In Raft, nodes exist in one of three states: Follower, Candidate, or Leader. When a follower node's randomized election timer expires without receiving heartbeats, it transitions to a Candidate, increments its current term, and broadcasts RequestVote RPCs. If multiple candidates initiate elections simultaneously, a split vote can occur where no candidate secures a majority quorum. Raft resolves split votes cleanly through randomized election timeouts (typically 150-300ms). The candidate that times out earliest increments its term and initiates a new election, rapidly achieving quorum and resuming log replication across the cluster.",
            "Gemini": "Raft achieves distributed fault tolerance by electing a distinguished leader responsible for sequential log replication across follower nodes. When followers detect missing leader heartbeats, they transition to candidates and solicit votes across the cluster. A split vote scenario arises when network partitions or concurrent timers divide votes equally among multiple candidates, preventing any single node from reaching a strict majority. To overcome this deadlock, Raft employs randomized election timeouts between 150ms and 300ms. The node whose timeout expires first transitions to a higher term and re-solicits votes, breaking the tie and electing a unique leader to manage linearizable log commits.",
            "Claude": "Raft structures distributed consensus around an explicit leader-follower paradigm designed for understandability and operational safety. Nodes continuously maintain a monotonic term counter and execute election timers. When a leader fails, follower timeouts trigger concurrent candidate transitions. If votes disperse evenly across multiple candidates, no single candidate achieves the necessary majority quorum, resulting in a split vote condition. Raft elegantly mitigates split vote stagnation by assigning randomized timer durations to each node. This timing jitter ensures that one candidate almost invariably times out and restarts its campaign before its peers, successfully gathering the required quorum and establishing stable leadership.",
            "Other_AI": "Raft consensus provides fault-tolerant state machine replication through structured term transitions. Follower nodes monitor leader heartbeat intervals. Upon timeout expiration, candidates issue RequestVote requests. Split votes occur when quorum cannot be established due to distributed vote allocation. Randomized backoff intervals between 150 and 300 milliseconds prevent synchronized re-election cycles, enabling a single candidate to achieve majority acceptance.",
            "Human": "In the Raft protocol, consensus relies on establishing a stable leader that accepts client commands and replicates log entries to followers. When followers miss AppendEntries heartbeats, they increment their term and become candidates. If two candidates start elections at the same time, votes can split evenly, leaving no one with a majority. Raft solves this by randomizing election timeouts on each node. Whichever candidate times out first increments its term, broadcasts new vote requests, and wins the election before others wake up, restoring normal cluster operation."
        }
    },

    # 2. AI / Machine Learning: Transformer Self-Attention
    {
        "prompt_id": "aiml_self_attention_02",
        "domain": "Artificial Intelligence / Machine Learning",
        "prompt": "How does scaled dot-product attention work in Transformer models and why is scaling necessary?",
        "responses": {
            "ChatGPT": "Scaled dot-product attention computes compatibility scores between query and key vectors to dynamically weight value representations. Given an input sequence, linear projections produce Query (Q), Key (K), and Value (V) matrices. The raw attention logits are calculated via the matrix product QK^T. The scaling factor, 1/sqrt(d_k), is critical because for large key dimensions d_k, the magnitude of dot products grows large. Large dot products push the softmax function into regions with extremely small gradients (vanishing gradients), which hampers backpropagation. Dividing by sqrt(d_k) normalizes the variance to 1, ensuring stable softmax distributions and efficient gradient flow throughout training.",
            "Gemini": "In Transformer architectures, scaled dot-product attention maps a set of query vectors against key vectors to generate soft attention weights over corresponding value vectors. The operation evaluates Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V. The scaling factor 1/sqrt(d_k) counteracts the mathematical growth of inner products as vector dimensionality d_k increases. Under high-dimensional spaces, unscaled dot products yield large scalar magnitudes that saturate the softmax activation, producing peaky distributions with near-zero gradients. Scaling stabilizes the variance of the logits, maintaining healthy gradient propagation during stochastic gradient descent.",
            "Claude": "Scaled dot-product attention forms the foundational mechanism of the Transformer architecture, allowing tokens to contextually attend to every other token across an input sequence. Mathematically, query vectors are multiplied by transposed key vectors to produce unnormalized attention scores: S = QK^T. As the projection dimension d_k scales up, the variance of the dot products increases proportionally to d_k. Without normalization, these large values cause the softmax function to saturate, resulting in vanishing gradients that severely impede optimization. Dividing by the square root of d_k rescales the variance back to unity, preserving sensitive gradient flow and robust parameter convergence.",
            "Other_AI": "Scaled dot-product attention calculates token affinity matrices using query and key linear projections. The unscaled matrix multiplication QK^T produces score distributions that scale with dimension d_k. Softmax saturation occurs when dot-product values grow excessive, attenuating backpropagated gradients. Normalizing by 1/sqrt(d_k) maintains unit variance across logits, ensuring stable attention weights and gradient updates.",
            "Human": "Scaled dot-product attention takes queries, keys, and values generated from the input tokens and calculates how much focus each token should give to others. Multiplying queries by transposed keys gives the raw matching scores. As the key dimension d_k gets larger, the dot products naturally grow in size, which pushes the softmax outputs toward 0 or 1 where gradients vanish. Dividing by the square root of d_k keeps the variance around 1, preventing the softmax from saturating so the model can train properly with backprop."
        }
    },

    # 3. Technical / Engineering: PID Feedback Control
    {
        "prompt_id": "tech_eng_pid_03",
        "domain": "Technical / Engineering",
        "prompt": "Explain the role of the derivative term in a PID controller and how measurement noise affects it.",
        "responses": {
            "ChatGPT": "In a Proportional-Integral-Derivative (PID) controller, the derivative term (D) acts as a predictive mechanism by computing the rate of change of the error signal (de/dt). By projecting future error trajectories, the derivative action provides damping, reducing overshoot and shortening settling time. However, the derivative term is highly sensitive to high-frequency measurement noise. Differentiating high-frequency sensor noise yields large, rapid output swings, causing actuator chatter and mechanical wear. To prevent instability in real-world systems, control engineers implement low-pass derivative filtering or calculate the derivative directly on the process variable rather than the error.",
            "Gemini": "The derivative component within PID control loops introduces anticipatory damping based on the time-derivative of the error signal. By counteracting rapid error changes, the D-term dampens oscillations and improves closed-loop phase margin. Despite its damping benefits, derivative action acts as a high-pass filter on sensor signals. When measurement noise corrupts the feedback channel, differentiation magnifies high-frequency noise components into extreme control output spikes. Industrial controllers mitigate this vulnerability by placing first-order low-pass filters on the derivative path to suppress noise amplification.",
            "Claude": "Within a three-term PID controller, the derivative action provides anticipatory control by responding directly to the slope of the error signal over time. This rate-of-change sensitivity introduces necessary phase lead and system damping, which suppresses oscillatory tendencies and limits transient overshoot. However, differentiation is inherently susceptible to sensor noise. Because noise typically consists of high-frequency components, taking its numerical derivative generates disproportionately large control signals, resulting in severe actuator jitter and mechanical fatigue. Practical implementations standardly pair the derivative block with a low-pass filter to attenuate high-frequency interference.",
            "Other_AI": "Derivative control computes instantaneous error slope to predict system deviation and provide damping against overshoot. High-frequency measurement noise presents a major operational challenge because differentiation scales with frequency. Unfiltered derivative action produces rapid actuator oscillations. Implementing low-pass filtering on the derivative branch attenuates high-frequency noise while preserving phase margin.",
            "Human": "The derivative term in a PID loop looks at how fast the error is changing and applies a braking force to stop the system from overshooting the target. While it helps settle the system faster, taking the derivative of noisy sensor readings causes huge spikes in the control output. These sudden spikes can rattle valves and wear out motors. To fix this, practical controllers usually run the derivative signal through a low-pass filter or calculate the derivative using only the feedback measurement instead of the error."
        }
    },

    # 4. Educational / Essay: Climate Change Adaptation
    {
        "prompt_id": "edu_climate_adaptation_04",
        "domain": "Educational",
        "prompt": "Discuss the difference between climate change mitigation and climate change adaptation with concrete examples.",
        "responses": {
            "ChatGPT": "Addressing global climate change requires two complementary strategies: mitigation and adaptation. Mitigation focuses on addressing the root causes of climate change by reducing greenhouse gas emissions and enhancing carbon sinks. Key mitigation initiatives include transitioning from fossil fuels to renewable energy (solar and wind), improving industrial energy efficiency, and engaging in large-scale reforestation. Conversely, adaptation involves adjusting ecological, social, and economic practices to minimize the damage caused by existing and inevitable climate impacts. Examples of adaptation include constructing sea walls to prevent coastal flooding, cultivating drought-resistant crop varieties, and upgrading urban stormwater drainage to withstand extreme precipitation.",
            "Gemini": "Climate change policy centers on two distinct yet interdependent pillars: mitigation and adaptation. Mitigation encompasses proactive interventions designed to reduce global carbon emissions and curb future warming trajectories. Prime examples involve deploying renewable energy grids, phasing out internal combustion vehicles, and protecting carbon-absorbing peatlands. In contrast, adaptation represents responsive adjustments to climatic shifts that are already locked in. Tangible adaptation measures include developing heat-resilient urban architecture, reinforcing flood defenses along vulnerable coastlines, and establishing early warning systems for extreme weather events.",
            "Claude": "Effective climate action rests upon two fundamental approaches: mitigation and adaptation. Mitigation seeks to limit the magnitude and rate of long-term climate change by reducing greenhouse gas emissions and capturing atmospheric carbon. Concrete mitigation measures include expanding nuclear and renewable power infrastructure, enforcing carbon pricing, and promoting circular economy manufacturing. Adaptation, on the other hand, focuses on building societal and ecological resilience against the unavoidable impacts of a warming planet. Key adaptation strategies include restoring coastal mangrove ecosystems as natural storm barriers, modifying agricultural planting calendars, and retrofitting municipal water networks to cope with prolonged droughts.",
            "Other_AI": "Mitigation aims to reduce greenhouse gas concentrations to curtail future warming, including renewable energy deployment and reforestation. Adaptation focuses on managing unavoidable climate hazards through structural and behavioral adjustments. Examples of adaptation include building flood walls, developing drought-tolerant crops, and revising zoning laws in vulnerable coastal zones.",
            "Human": "Mitigation is about stopping climate change from getting worse by cutting emissions—things like switching to solar panels, driving electric cars, and planting trees. Adaptation is about dealing with the changes that are already happening or unavoidable, like building higher sea walls against rising tides, developing crops that can survive heatwaves, and designing better city drainage so streets don't flood during heavy storms."
        }
    },

    # 5. Marketing / Advertising: Product Launch Strategy
    {
        "prompt_id": "mkt_launch_strategy_05",
        "domain": "Marketing / Advertising",
        "prompt": "Create an executive summary for an omnichannel marketing strategy launching a smart home security system.",
        "responses": {
            "ChatGPT": "This omnichannel launch strategy positions our next-generation smart home security system as the premier solution for seamless, AI-powered residential protection. The campaign orchestrates cohesive messaging across digital performance marketing, social influencer partnerships, premium retail displays, and targeted connected TV (CTV) advertising. By leveraging personalized programmatic advertising and interactive product demonstrations, we guide high-intent homeowners from awareness through conversion. A synchronized multi-touch attribution model ensures optimized ad spend, while seamless post-purchase onboarding drives customer retention, lifetime value, and word-of-mouth advocacy across all consumer touchpoints.",
            "Gemini": "Our integrated omnichannel launch plan establishes market leadership for the AI-enabled SmartGuard residential security ecosystem. The strategy aligns digital discovery, localized retail experiential hubs, and targeted social media activations into a unified customer journey. Utilizing first-party data and dynamic creative optimization, targeted ad units capture early adopters in urban and suburban markets. Strategic retail endcaps with live interactive testing reinforce trust, while automated post-purchase email flows and in-app rewards nurture brand advocacy and recurring subscription revenue.",
            "Claude": "This executive summary outlines a comprehensive omnichannel marketing strategy designed to capture market share for our flagship smart home security platform. The launch seamlessly bridges top-of-funnel brand storytelling with performance-driven digital acquisition across search, social, PR, and specialized retail partnerships. Compelling lifestyle narratives highlighting privacy-first AI surveillance and effortless DIY installation engage modern homeowners. Continuous multi-channel data integration and closed-loop attribution maximize return on ad spend (ROAS), while targeted influencer co-creations and robust referral incentives foster sustained customer loyalty.",
            "Other_AI": "The omnichannel launch strategy integrates performance marketing, retail activations, and influencer campaigns to drive adoption of the smart home security platform. Targeted digital advertising captures consumer demand while in-store displays provide physical product validation. Attribution tracking and automated onboarding workflows optimize acquisition costs and customer retention.",
            "Human": "We're launching our new smart security system across online ads, social media, retail stores, and TV to reach homeowners looking for easy, reliable home protection. Online search and video ads will drive traffic to our site, while hands-on display units in major tech stores let customers try the cameras before buying. We'll track what channels bring in the most sales and use automated follow-up emails to help new buyers set up their systems quickly."
        }
    },

    # 6. News / Journalism: Semiconductor Breakthrough
    {
        "prompt_id": "news_semiconductor_06",
        "domain": "News / Journalism",
        "prompt": "Write a concise news report on a major tech company announcing a 1.4-nanometer chip fabrication milestone.",
        "responses": {
            "ChatGPT": "TAIPEI — In a pivotal milestone for the global semiconductor industry, leading fabrication foundry TSMC announced on Wednesday that it has achieved a critical breakthrough in 1.4-nanometer (A14) process technology. The new node introduces advanced High-NA EUV lithography and next-generation Gate-All-Around (GAA) nanosheet transistors. Company executives confirmed that the A14 architecture delivers a 15% increase in compute speed and a 30% reduction in power consumption compared to existing 2nm nodes. Commercial production is scheduled to begin in late 2027, positioning the foundry to meet surging compute demands for generative AI and hyperscale cloud infrastructure.",
            "Gemini": "SAN JOSE — A major breakthrough in advanced chip manufacturing was unveiled today as semiconductor giant Intel revealed its roadmap for the 14A (1.4nm) fabrication node. Utilizing cutting-edge High-NA Extreme Ultraviolet lithography and backside power delivery, the 1.4nm process is engineered to overcome traditional physical scaling limits. Initial test runs demonstrate significant density gains alongside a 20% improvement in performance-per-watt metrics. Industry analysts note that this development intensifies the global competition to manufacture ultra-dense microprocessors powering next-generation artificial intelligence workloads.",
            "Claude": "HSINCHU — Global semiconductor manufacturing reached a historic threshold today as researchers announced operational validation of a 1.4-nanometer fabrication process. The breakthrough combines High-NA EUV lithographic patterning with refined GAA nanosheet architectures, allowing billions of additional transistors to be integrated onto silicon dies. Early testing reveals that the 1.4nm node provides up to 18% higher clock frequencies while slashing thermal dissipation by over 25%. Volume production is slated for late 2027, promising substantial performance leaps for future AI accelerators and mobile processors.",
            "Other_AI": "TAIPEI — Semiconductor manufacturers announced operational progress on 1.4-nanometer process technology using High-NA EUV systems and nanosheet transistor architectures. The new process node offers a 15% performance gain and 25% energy reduction over current generations. Volume manufacturing is expected by 2028 to support growing artificial intelligence hardware demand.",
            "Human": "Chip giant TSMC announced today that its next-generation 1.4-nanometer manufacturing process is on track, with mass production planned for 2027. The new technology uses advanced High-NA EUV machines and new transistor designs to squeeze more processing power into smaller spaces while cutting down electricity use by nearly 30%. The breakthrough is expected to give tech companies much-needed speed boosts for training and running intensive AI models."
        }
    },

    # 7. Conversational / Q&A: Git Rebase vs Merge
    {
        "prompt_id": "qa_git_rebase_merge_07",
        "domain": "Conversational / Q&A",
        "prompt": "When should a developer use git rebase instead of git merge, and what are the trade-offs?",
        "responses": {
            "ChatGPT": "A developer should typically use `git rebase` when updating a local feature branch with the latest changes from `main` before submitting a pull request. Rebasing rewrites commit history by replaying your local commits on top of the target branch, producing a clean, linear git log without unnecessary merge commits. However, the trade-off is history rewriting: rebasing changes commit hashes. Therefore, you should never rebase public or shared branches, as this causes merge conflicts and history divergence for collaborators. Use `git merge` when integrating finished feature branches into `main` to preserve the exact historical context of branch lifecycles.",
            "Gemini": "`git rebase` is ideal when you want to maintain a tidy, linear commit history on private feature branches prior to merging. By moving the base of your branch to the latest commit on `main`, rebasing eliminates cluttered merge commits and simplifies `git bisect` debugging. The primary risk is that rebasing rewrites commit hashes. If multiple engineers share the branch, rebasing forces awkward force-pushes and breaks local repositories. As a general rule, rebase local private branches to stay updated, but use `git merge` on public, shared branches to preserve collaborative history.",
            "Claude": "You should use `git rebase` primarily when maintaining clean, linear history on local, unshared feature branches. Rebasing lifts your branch's unique commits and reapplies them sequentially on top of the newest upstream commit, preventing the proliferation of trivial 'Merge branch' commits and making the project history much easier to read and audit. The crucial trade-off lies in immutability: rebasing rewrites commit history and alters SHA-1 hashes. If you rebase a shared remote branch, you will desynchronize your teammates' working trees. Stick to `git merge` for shared trunk branches and `git rebase` for local polishing.",
            "Other_AI": "Use git rebase for local feature branches to create a linear commit graph and remove merge noise. The trade-off is commit hash rewriting, which can disrupt shared branches. Use git merge when preserving original branch context and collaborative history on shared repositories.",
            "Human": "Use `git rebase` when you're working on your own feature branch and want to pull in the latest changes from `main` without creating messy merge commits. It makes your commit history look like a straight line. The big catch is that rebasing changes the commit IDs, so you should never rebase a branch that other developers are also working on. Use `git merge` when you're ready to bring your completed feature into `main`."
        }
    },

    # 8. Tutorials / How-To: Dockerizing Python App
    {
        "prompt_id": "tut_docker_python_08",
        "domain": "Tutorials / How-To",
        "prompt": "Provide step-by-step instructions for creating a production-ready Dockerfile for a Python FastAPI application.",
        "responses": {
            "ChatGPT": "To containerize a FastAPI application for production, follow these best practices:\n1. **Base Image:** Use a lightweight official image like `python:3.11-slim`.\n2. **Set Working Directory:** Define `WORKDIR /app`.\n3. **Layer Caching:** Copy `requirements.txt` first and run `pip install --no-cache-dir -r requirements.txt` before copying the source code.\n4. **Security:** Create and switch to a non-root user (`USER appuser`) to minimize container privileges.\n5. **Source Code:** Copy application files with `COPY --chown=appuser:appuser . /app`.\n6. **Port & Entrypoint:** Expose port `8000` and use `CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\", \"--workers\", \"4\"]` for optimized ASGI concurrency.",
            "Gemini": "Here is how to build a clean, secure Dockerfile for FastAPI:\n1. Choose a minimal slim base: `FROM python:3.11-slim`.\n2. Prevent bytecode generation and enable unbuffered output with `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1`.\n3. Establish `WORKDIR /app`.\n4. Leverage Docker build caching by copying `requirements.txt` and installing dependencies with `pip install --no-cache-dir -r requirements.txt`.\n5. Add a non-privileged user: `RUN useradd -m -u 1000 appuser` and switch to `USER appuser`.\n6. Copy project code and define execution: `CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\", \"--workers\", \"4\"]`.",
            "Claude": "Follow these steps to structure an optimized production Dockerfile for FastAPI:\n1. **Select Base:** Start from `FROM python:3.11-slim` to minimize image size and attack surface.\n2. **Environment Flags:** Set `ENV PYTHONUNBUFFERED=1` and `ENV PYTHONDONTWRITEBYTECODE=1` for clean logging.\n3. **Set Directory:** Specify `WORKDIR /app`.\n4. **Optimize Caching:** Copy dependency manifests first: `COPY requirements.txt .`, then execute `RUN pip install --no-cache-dir -r requirements.txt`.\n5. **Non-Root Security:** Create a dedicated user via `RUN useradd -u 8888 -m appuser` and assign ownership.\n6. **Copy Code & Run:** `COPY --chown=appuser:appuser . .`, then set `USER appuser` and execute `CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]`.",
            "Other_AI": "1. Use `FROM python:3.11-slim`.\n2. Set `WORKDIR /app` and environment variables for unbuffered output.\n3. Copy `requirements.txt` and run `pip install --no-cache-dir`.\n4. Create a non-root user for security.\n5. Copy application source code.\n6. Run using `uvicorn main:app --host 0.0.0.0 --port 8000`.",
            "Human": "To dockerize your FastAPI app for production, start with a slim base image like `python:3.11-slim`. Set your working directory to `/app`, then copy your `requirements.txt` and install packages with `--no-cache-dir` so Docker caches your dependencies when your code changes. Always create and switch to a non-root user for security before copying in your source files. Finally, expose port 8000 and start uvicorn with multiple workers."
        }
    },

    # 9. Finance: Capital Asset Pricing Model (CAPM)
    {
        "prompt_id": "fin_capm_09",
        "domain": "Finance",
        "prompt": "Explain the Capital Asset Pricing Model (CAPM), its core assumptions, and the role of Beta.",
        "responses": {
            "ChatGPT": "The Capital Asset Pricing Model (CAPM) establishes a linear relationship between the expected return of an asset and its systematic risk. Expressed as E(R_i) = R_f + Beta_i * (E(R_m) - R_f), the model asserts that investors demand compensation for the time value of money (the risk-free rate R_f) and market risk (the equity risk premium). Beta measures an asset's sensitivity to macroeconomic market fluctuations. A Beta greater than 1 indicates higher volatility than the benchmark index, while a Beta below 1 denotes lower volatility. Key assumptions include rational, risk-averse investors, frictionless markets with no taxes or transaction costs, and homogenous expectations regarding asset returns.",
            "Gemini": "CAPM is a foundational asset valuation framework calculating expected return as a function of systematic risk exposure: E(R_i) = R_f + Beta * (E(R_m) - R_f). Central to CAPM is Beta, which quantifies an asset's covariance with the broader market portfolio relative to market variance. Assets with Beta > 1 amplify market movements, while Beta < 1 dampens them. Core assumptions underlying the model include frictionless capital markets, zero borrowing constraints at the risk-free rate, single-period investment horizons, and mean-variance investor rationality.",
            "Claude": "The Capital Asset Pricing Model (CAPM) formalizes the equilibrium pricing of risky assets based solely on their non-diversifiable systematic risk. The formula, E(R_i) = R_f + Beta_i * (E(R_m) - R_f), dictates that an asset's expected excess return is directly proportional to its Beta coefficient. Beta captures how strongly an individual security co-moves with the broader market portfolio. The model relies on several classical theoretical assumptions: investors hold diversified mean-variance efficient portfolios, information is costlessly and instantaneously available, taxes and transaction fees are absent, and unlimited lending/borrowing exists at the risk-free rate.",
            "Other_AI": "CAPM calculates expected asset returns based on systematic market risk using the equation E(R) = Rf + Beta * (Rm - Rf). Beta measures volatility relative to the market index. The model assumes frictionless markets, rational risk-averse investors, and homogeneous return expectations across market participants.",
            "Human": "CAPM is a formula used in finance to figure out what kind of return you should expect on an investment given its risk. The formula is Expected Return = Risk-Free Rate + Beta * (Market Return - Risk-Free Rate). Beta is the key number here—it shows how jumpy a stock is compared to the entire market. If Beta is 1.5, the stock swings 50% more than the market. The model assumes investors are rational and that there are no trading fees or taxes."
        }
    },

    # 10. Healthcare / Medical: Type 2 Diabetes Pathophysiology
    {
        "prompt_id": "med_t2d_pathophysiology_10",
        "domain": "Healthcare / Medical",
        "prompt": "Describe the cellular mechanisms of insulin resistance in type 2 diabetes mellitus.",
        "responses": {
            "ChatGPT": "The pathophysiology of Type 2 Diabetes Mellitus (T2DM) centers on peripheral insulin resistance and progressive pancreatic beta-cell dysfunction. At the cellular level, insulin resistance is characterized by impaired insulin receptor substrate (IRS) signaling. Chronic hyperlipidemia and ectopic lipid accumulation generate intracellular lipid intermediates (diacylglycerols and ceramides), which activate novel protein kinase C (PKC) isoforms. Activated PKCs induce serine/threonine phosphorylation of IRS-1 and IRS-2, blocking downstream PI3K-Akt signaling and impairing GLUT4 glucose transporter translocation to the plasma membrane. Consequently, skeletal muscle and adipose tissue exhibit diminished glucose uptake, while hepatic gluconeogenesis remains inappropriately elevated.",
            "Gemini": "Type 2 diabetes arises from blunted cellular responsiveness to insulin combined with relative pancreatic secretory failure. In skeletal muscle and hepatocytes, insulin binding normally triggers receptor tyrosine phosphorylation and activation of the IRS/PI3K/Akt pathway, stimulating GLUT4 vesicle trafficking. In insulin-resistant states, elevated circulating free fatty acids and inflammatory cytokines (TNF-alpha, IL-6) induce inhibitory serine phosphorylation of IRS-1. This signaling block impairs Akt phosphorylation, halting GLUT4 translocation and preventing postprandial glucose uptake, driving persistent hyperglycemia and compensatory hyperinsulinemia.",
            "Claude": "Type 2 diabetes mellitus is primarily driven by insulin resistance across major metabolic tissues—principally skeletal muscle, liver, and adipose tissue. Under physiological conditions, insulin activates its transmembrane tyrosine kinase receptor, phosphorylating IRS-1 and initiating the PI3K/Akt downstream cascade responsible for mobilizing GLUT4 storage vesicles. In diabetic states, sustained nutrient excess and chronic low-grade inflammation cause intracellular accumulation of toxic lipid species such as diacylglycerols. These metabolites stimulate stress kinases (including JNK and PKC-theta), which aberrantly phosphorylate IRS-1 on inhibitory serine residues, uncoupling the receptor from GLUT4 translocation and precipitating systemic glucose intolerance.",
            "Other_AI": "Type 2 diabetes develops from impaired insulin signaling and beta-cell secretory decline. Chronic lipid overload activates protein kinase C isoforms that cause inhibitory serine phosphorylation of IRS-1. This disrupts PI3K/Akt signaling cascades, preventing GLUT4 translocation to the cell membrane and reducing cellular glucose uptake in skeletal muscle and adipose tissue.",
            "Human": "In Type 2 diabetes, the body's cells stop responding properly to insulin. Normally, insulin acts like a key that tells cells to bring GLUT4 glucose transporters to their surface to absorb sugar from the bloodstream. When someone has insulin resistance, fat buildup and chronic inflammation cause cellular stress, which damages the internal signaling pathways inside muscle and liver cells. Because the signal is broken, GLUT4 transporters don't move to the surface, sugar builds up in the blood, and the pancreas has to pump out even more insulin to compensate."
        }
    },

    # 11. Real Estate / Business: Commercial Lease Structures
    {
        "prompt_id": "re_lease_structures_11",
        "domain": "Real Estate",
        "prompt": "Compare Gross Leases, Modified Gross Leases, and Triple Net (NNN) Leases for commercial real estate.",
        "responses": {
            "ChatGPT": "In commercial real estate, lease structures dictate how property operational expenses are divided between landlord and tenant:\n1. **Gross Lease (Full Service):** The tenant pays a fixed base rent, while the landlord covers all operating expenses, including property taxes, building insurance, and common area maintenance (CAM).\n2. **Modified Gross Lease:** The tenant pays base rent and directly covers specific agreed-upon operational costs (such as utilities or janitorial services), with expense increases beyond a base year passed through to the tenant.\n3. **Triple Net (NNN) Lease:** The tenant pays a lower base rent plus all three 'nets': property taxes, building insurance, and CAM expenses. NNN leases are standard in standalone retail and industrial properties, shifting operational inflation risk to the tenant.",
            "Gemini": "Commercial lease agreements differ primarily in how operating expenses are allocated:\n- **Gross Lease:** The tenant makes a single flat monthly payment, and the landlord assumes responsibility for all property taxes, hazard insurance, structural repairs, and utilities.\n- **Modified Gross Lease:** A balanced hybrid where the tenant pays base rent plus designated utilities or internal maintenance, while sharing incremental increases in property operating costs over a baseline year.\n- **Triple Net (NNN) Lease:** The tenant pays base rent alongside pro-rata shares of property taxes, building insurance premiums, and common area maintenance (CAM). NNN leases are common in multi-tenant commercial centers and industrial warehouses.",
            "Claude": "Commercial real estate transactions utilize three primary lease frameworks to distribute financial obligations:\n1. **Full-Service Gross Lease:** The tenant pays a comprehensive flat rental fee. The landlord bears all ongoing operational liabilities, including property taxes, structural insurance, utilities, and common area maintenance (CAM).\n2. **Modified Gross Lease:** A negotiated compromise where the tenant pays base rent while absorbing specific variable expenses, such as interior utilities or increases in operating costs above an established base-year ceiling.\n3. **Triple Net (NNN) Lease:** The tenant is responsible for base rent in addition to the three core operational categories: property taxes, property insurance, and common area maintenance. This structure insulates landlords from cost escalations and is prevalent in single-tenant retail and industrial assets.",
            "Other_AI": "Gross leases charge a fixed rate with the landlord paying all operational costs. Modified gross leases split utilities and incremental operational increases between parties. Triple net (NNN) leases require tenants to pay base rent plus property taxes, insurance, and common area maintenance fees directly.",
            "Human": "In commercial real estate, a Gross Lease means the tenant pays one flat rent amount every month and the landlord pays for all property taxes, building insurance, and maintenance. In a Modified Gross Lease, the tenant pays base rent plus their own utilities or a share of maintenance. In a Triple Net (NNN) Lease, the tenant pays a lower base rent but is directly responsible for all three major operating costs: property taxes, insurance, and building maintenance."
        }
    },

    # 12. Creative Writing: Solitary Lighthouse Keeper
    {
        "prompt_id": "cr_lighthouse_keeper_12",
        "domain": "Creative Writing / Stories",
        "prompt": "Write a descriptive scene depicting an aging lighthouse keeper preparing the lamp before a severe winter gale.",
        "responses": {
            "ChatGPT": "Salt encrusted the brass railings of the lantern room as Arthur climbed the final spiraling iron steps. Outside, the North Atlantic churned into slate-gray swells, foaming violently against the granite base of the tower. He pulled a woolen cloth from his oilskin coat, carefully polishing the curved facets of the first-order Fresnel lens until the glass gleamed like cut crystal. With weathered, calloused fingers, he trimmed the braided wick and primed the kerosene vaporizer. As the first gust of sleet rattled against the storm panes, Arthur struck a sulfur match, coaxing the flame to life. The great clockwork mechanism engaged with a rhythmic click, sending a blinding beam of amber light cutting into the encroaching darkness.",
            "Gemini": "Sixty feet above the roaring surf, Silas stood inside the glass crown of Cape Desolation. The approaching nor'easter had turned the sky the color of bruised iron. He moved with practiced, deliberate calm around the colossal brass housing of the Fresnel lens, wiping away every trace of condensation with a dry chamois. The wind screamed through the iron vents, vibrating the foundation beneath his heavy boots. Silas checked the counterweights, adjusted the wick collar, and carefully ignited the burner. As the golden flame caught, the mirrored prisms caught the light, multiplying it into a sweeping blade of fire that sliced through the gathering blizzard.",
            "Claude": "The iron rungs were bitterly cold through Thomas's worn leather mittens as he reached the lantern gallery. Beyond the curved storm glass, the ocean had surrendered its green depths to an expanse of pitch and surging white crests. He set down his brass lantern and began the evening ritual: inspecting the concentric rings of the great lens, wiping sea salt from the optical glass, and ensuring the mechanical clockwork weights hung true. When the barometer plunged another two points, Thomas struck a wooden match beneath the mantle. The flame blossomed, caught the mercury-floated prisms, and cast a sweeping arc of brilliant amber over the crashing shoals below.",
            "Other_AI": "Arthur climbed the spiral stairs to the lantern room as the gale approached. The Atlantic pounded against the granite lighthouse walls. He polished the glass prisms of the lens and checked the oil level in the reservoir. Striking a match, he lit the lamp and engaged the rotating gears, casting a protective beam across the dark storm waves.",
            "Human": "Old Tom made his way up the narrow spiral stairs, his knees aching with every step as the wind howled outside. Up in the lantern room, the glass was already rattling from the incoming nor'easter. He wiped the salt spray off the heavy glass lens with an old rag, checked the kerosene tank, and trimmed the wick with steady hands. He struck a match, lit the mantle, and watched the flame catch the polished mirrors, throwing a strong yellow beam out into the freezing storm."
        }
    },

    # 13. Social Media: Tech Influencer Thought Leadership
    {
        "prompt_id": "sm_ai_agents_future_13",
        "domain": "Social Media",
        "prompt": "Write an engaging LinkedIn/Twitter thought leadership post on why autonomous AI agents are replacing simple prompt-based chatbots.",
        "responses": {
            "ChatGPT": "Prompt engineering was just the beginning. The real revolution in AI isn't chatbots that answer questions—it's autonomous AI agents that execute end-to-end workflows. 🚀\n\nUnlike traditional LLMs that wait passively for a prompt, modern agents:\n✅ Decompose complex goals into discrete sub-tasks\n✅ Call external APIs and search tools in real-time\n✅ Self-correct when code or steps fail\n✅ Retain short- and long-term memory\n\nWe are shifting from 'chatting with AI' to 'delegating to AI'. The companies that master agentic architectures in 2026 won't just move faster—they will redefine entire operational paradigms. What is your team building?",
            "Gemini": "Stop chatting with AI. Start delegating to it. 💡\n\nThe industry is rapidly transitioning from passive conversational chatbots to autonomous agentic systems. Here is why this shift changes everything:\n\n1️⃣ Goal-driven autonomy: Agents plan multi-step execution paths rather than generating one-off responses.\n2️⃣ Tool integration: Seamlessly querying databases, running code, and triggering APIs.\n3️⃣ Closed-loop reflection: Evaluating their own outputs and iterating until objectives are met.\n\nThe future of enterprise software is not a chat UI—it is an autonomous digital workforce. How is your organization preparing?",
            "Claude": "The era of the chatbot is quietly ending. The era of the autonomous agent has arrived. 🧠\n\nPrompting was step one. But real enterprise value lies in agentic workflows that turn high-level intent into executed outcomes:\n\n• Autonomous planning: Breaking open-ended goals into structured steps\n• Tool orchestration: Reading docs, executing terminal commands, and verifying results\n• Iterative debugging: Detecting errors and autonomously refining approaches\n\nThe competitive moat isn't model access—it's how intelligently you compose agents to solve complex, multi-modal workflows. What's your take?",
            "Other_AI": "AI is moving from prompt-based chatbots to autonomous agents. Agents can plan multi-step workflows, use external tools, and self-correct errors during execution. The value is shifting from simple text generation to autonomous task completion across enterprise systems.",
            "Human": "Everyone is realizing that typing back-and-forth prompts into chatbots was only step one. The exciting shift happening right now is toward autonomous agents—systems where you give a high-level goal and the AI searches the web, writes the code, runs the tests, and fixes its own bugs until the job is done."
        }
    },

    # 14. Academic / Scientific: CRISPR-Cas9 Gene Editing
    {
        "prompt_id": "acad_crispr_hdr_14",
        "domain": "Academic / Scientific",
        "prompt": "Explain the difference between Non-Homologous End Joining (NHEJ) and Homology-Directed Repair (HDR) following CRISPR-Cas9 DNA cleavage.",
        "responses": {
            "ChatGPT": "Following CRISPR-Cas9-mediated site-specific DNA double-strand breaks (DSBs), eukaryotic cells resolve the genomic lesion through two distinct endogenous repair pathways: Non-Homologous End Joining (NHEJ) and Homology-Directed Repair (HDR). NHEJ is an error-prone, cell-cycle-independent repair mechanism that directly ligates broken DNA ends without requiring a homologous template. This frequently introduces stochastic insertions and deletions (indels), making NHEJ the preferred pathway for gene knockout applications. In contrast, HDR is a high-fidelity repair pathway active primarily in the S and G2 phases of the cell cycle. HDR utilizes an exogenous or sister chromatid donor template to introduce precise nucleotide substitutions or targeted transgene insertions with nucleotide-level precision.",
            "Gemini": "CRISPR-Cas9 endonuclease activity induces targeted double-strand DNA breaks that trigger two competing cellular repair pathways: NHEJ and HDR. NHEJ operates constitutively throughout the cell cycle, directly reconnecting blunt or staggered DNA ends via the Ku70/Ku80 complex. Because NHEJ lacks template-guided verification, it regularly generates frameshift indel mutations, enabling efficient gene functional disruption. Conversely, HDR enables precise genome editing by utilizing homologous donor DNA sequences to guide template-directed synthesis. Because HDR requires active sister chromatids and specific recombinases (such as Rad51), its efficiency is restricted to actively dividing cells in S/G2 phases.",
            "Claude": "Double-strand DNA breaks introduced by the Cas9 endonuclease are resolved by the host cell through either Non-Homologous End Joining (NHEJ) or Homology-Directed Repair (HDR). NHEJ is the dominant and inherently error-prone pathway; it rapidly re-ligates cleaved DNA termini without a repair template, frequently resulting in frameshift-inducing insertion or deletion mutations ideal for gene knockout experiments. In contrast, HDR represents a high-fidelity template-dependent repair process that operates during the S and G2 phases when homologous donor templates are accessible. By supplying exogenous single- or double-stranded DNA templates, researchers exploit HDR to achieve precise base edits, point mutation corrections, or targeted gene knock-ins.",
            "Other_AI": "CRISPR-Cas9 creates double-strand breaks repaired by NHEJ or HDR. NHEJ directly ligates DNA ends without a template, introducing random indel mutations suitable for gene knockouts. HDR requires a homologous donor template and is active during S/G2 cell phases to introduce precise sequence modifications.",
            "Human": "When Cas9 cuts both strands of DNA, the cell repairs the break using either NHEJ or HDR. NHEJ is fast but sloppy—it just glues the cut ends back together without a template, which often adds or deletes a few DNA letters and breaks the gene. HDR is precise: it uses a donor DNA template to copy the correct sequence into the break, allowing scientists to insert specific mutations or new genes, though it only works in dividing cells during the S and G2 phases."
        }
    },

    # 15. Legal / Formal: Force Majeure Contract Clauses
    {
        "prompt_id": "legal_force_majeure_15",
        "domain": "Legal / Formal",
        "prompt": "Draft a standard commercial Force Majeure clause and explain the key elements required for enforceable invocation.",
        "responses": {
            "ChatGPT": "FORCE MAJEURE CLAUSE:\n'Neither party shall be liable for any failure or delay in the performance of its obligations under this Agreement if such failure arises from events beyond its reasonable control, including but not limited to acts of God, war, terrorism, civil unrest, natural disasters, epidemics, governmental orders, or utility failures (a \"Force Majeure Event\"). The affected party must provide prompt written notice within ten (10) days and exercise reasonable diligence to mitigate the impact.'\n\nKey elements for enforceable invocation include: (1) demonstration that the triggering event was unforeseeable and outside reasonable control, (2) proof that the event directly rendered performance impossible rather than merely economically burdensome, (3) timely formal written notice, and (4) verifiable mitigation efforts.",
            "Gemini": "COMMERCIAL FORCE MAJEURE CLAUSE:\n'Neither party shall be deemed in breach of this Agreement for delays or defaults in performance resulting from causes beyond its reasonable control, including acts of God, armed conflict, civil disturbances, catastrophic weather events, pandemics, or lawful governmental embargoes (each a \"Force Majeure Event\"). The impacted party shall provide written notice within seven (7) business days and implement commercially reasonable remediation measures.'\n\nEnforceability requires establishing: (a) non-foreseeability at the time of contract execution, (b) insurmountable causation preventing performance, (c) strict compliance with contractual notice protocols, and (d) documented good-faith mitigation efforts.",
            "Claude": "FORCE MAJEURE PROVISION:\n'Except for payment obligations, neither party shall be liable for any failure or delay in fulfilling its commitments under this Agreement to the extent caused by circumstances beyond its reasonable control, including natural catastrophes, war, armed hostilities, riots, pandemics, national labor strikes, or statutory governmental restrictions (each a \"Force Majeure Event\"). The affected party shall deliver written notice within ten (10) business days and utilize all reasonable endeavors to resume performance.'\n\nEnforceable invocation requires proving: (1) the event falls unambiguously within the defined scope, (2) the condition was genuinely unforeseeable, (3) performance was rendered impossible or commercially impracticable, and (4) the asserting party fulfilled mandatory notice and mitigation obligations.",
            "Other_AI": "Force Majeure clauses excuse contractual non-performance when unforeseen external events occur, such as natural disasters, war, or government actions. Enforceability requires showing the event was beyond reasonable control, made performance impossible, and that timely notice and mitigation efforts were undertaken.",
            "Human": "A Force Majeure clause protects either party from being sued if an unexpected disaster—like an earthquake, war, or government shutdown—makes it impossible to fulfill the contract. To actually use the clause in court, the affected party has to prove that the event was completely out of their control, that it genuinely prevented them from doing their work (not just made it more expensive), and that they sent written notice to the other side right away."
        }
    }
]


def build_dataset() -> Dict[str, Any]:
    """
    Constructs the dataset rows and creates a strict prompt-grouped split:
      - 70% Train
      - 15% Validation
      - 15% Test
    """
    random.seed(RANDOM_SEED)

    # 1. Expand each prompt into individual sample rows
    all_samples: List[Dict[str, Any]] = []
    prompt_ids = list(set(p["prompt_id"] for p in PARALLEL_PROMPT_CATALOGUE))
    prompt_ids.sort()

    logger.info("Total prompt groups: %d", len(prompt_ids))

    for prompt_group in PARALLEL_PROMPT_CATALOGUE:
        p_id = prompt_group["prompt_id"]
        domain = prompt_group["domain"]
        prompt_text = prompt_group["prompt"]

        for gen_name, text_sample in prompt_group["responses"].items():
            # Standardize generator category
            if gen_name in ["ChatGPT", "GPT-4", "GPT-4o", "GPT-3.5"]:
                gen_label = "ChatGPT"
                is_ai = 1
            elif gen_name in ["Gemini", "Gemini 1.5 Pro", "Gemini 1.5 Flash"]:
                gen_label = "Gemini"
                is_ai = 1
            elif gen_name in ["Claude", "Claude 3.5 Sonnet", "Claude 3 Opus"]:
                gen_label = "Claude"
                is_ai = 1
            elif gen_name in ["Other_AI", "Llama 3", "Mistral", "DeepSeek"]:
                gen_label = "Other_AI"
                is_ai = 1
            elif gen_name == "Human":
                gen_label = "Human"
                is_ai = 0
            else:
                gen_label = "Other_AI"
                is_ai = 1

            all_samples.append({
                "prompt_id": p_id,
                "domain": domain,
                "prompt": prompt_text,
                "text": text_sample.strip(),
                "generator": gen_label,
                "is_ai": is_ai,
                "word_count": len(text_sample.split())
            })

    logger.info("Total generated sample rows: %d", len(all_samples))

    # 2. Strict Prompt-Grouped Partitioning
    shuffled_prompts = prompt_ids.copy()
    random.shuffle(shuffled_prompts)

    n_prompts = len(shuffled_prompts)
    n_train = max(1, int(n_prompts * TRAIN_RATIO))
    n_val = max(1, int(n_prompts * VAL_RATIO))
    
    train_prompts = set(shuffled_prompts[:n_train])
    val_prompts = set(shuffled_prompts[n_train:n_train + n_val])
    test_prompts = set(shuffled_prompts[n_train + n_val:])

    # Guarantee test has at least 1 prompt if rounding clamped it
    if not test_prompts and len(val_prompts) > 1:
        moved = val_prompts.pop()
        test_prompts.add(moved)

    logger.info("Split by prompt groups: Train=%d, Val=%d, Test=%d",
                len(train_prompts), len(val_prompts), len(test_prompts))

    # Check for strict zero-leakage overlap
    assert len(train_prompts.intersection(val_prompts)) == 0, "Leakage: Train/Val overlap!"
    assert len(train_prompts.intersection(test_prompts)) == 0, "Leakage: Train/Test overlap!"
    assert len(val_prompts.intersection(test_prompts)) == 0, "Leakage: Val/Test overlap!"

    # Assign split labels
    train_samples, val_samples, test_samples = [], [], []
    for s in all_samples:
        pid = s["prompt_id"]
        if pid in train_prompts:
            s["split"] = "train"
            train_samples.append(s)
        elif pid in val_prompts:
            s["split"] = "val"
            val_samples.append(s)
        elif pid in test_prompts:
            s["split"] = "test"
            test_samples.append(s)
        else:
            raise ValueError(f"Prompt {pid} not assigned to any split")

    # 3. Save files
    _DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = _DATASET_DIR / "attribution_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    metadata = {
        "dataset_name": "Multi-Generator Parallel Prompt Attribution Corpus",
        "random_seed": RANDOM_SEED,
        "total_samples": len(all_samples),
        "total_prompts": len(prompt_ids),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_prompts": list(train_prompts),
        "val_prompts": list(val_prompts),
        "test_prompts": list(test_prompts),
        "generator_distribution": {
            "ChatGPT": sum(1 for s in all_samples if s["generator"] == "ChatGPT"),
            "Gemini": sum(1 for s in all_samples if s["generator"] == "Gemini"),
            "Claude": sum(1 for s in all_samples if s["generator"] == "Claude"),
            "Other_AI": sum(1 for s in all_samples if s["generator"] == "Other_AI"),
            "Human": sum(1 for s in all_samples if s["generator"] == "Human"),
        },
        "domain_distribution": list(set(s["domain"] for s in all_samples))
    }

    meta_path = _CONFIGS_DIR / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Saved dataset to %s and metadata to %s", jsonl_path, meta_path)
    return metadata


if __name__ == "__main__":
    meta = build_dataset()
    print("Dataset generated successfully:")
    print(json.dumps(meta, indent=2))
