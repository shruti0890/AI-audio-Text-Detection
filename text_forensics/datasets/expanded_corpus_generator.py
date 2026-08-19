"""
text_forensics/datasets/expanded_corpus_generator.py

Catalog of multi-domain paired Human and AI passages covering all 18 domains:
  1. Technical / Engineering
  2. Computer Science
  3. Artificial Intelligence / Machine Learning
  4. Academic / Scientific
  5. Educational
  6. Social Media
  7. Marketing / Advertising
  8. Real Estate
  9. Blogs / Articles
  10. News / Journalism
  11. Conversational / Q&A
  12. Product Reviews
  13. Finance
  14. Healthcare / Medical
  15. Legal / Formal
  16. Creative Writing / Stories
  17. Tutorials / How-To
  18. General Informational Writing
"""

from typing import Any, Dict, List

DOMAIN_DATA_PACKS: List[Dict[str, Any]] = [
    # 1. Technical / Engineering
    {
        "domain": "Technical / Engineering",
        "topic": "Feedback Control Stability",
        "prompt_group": "tech_eng_pid",
        "human_text": "In automatic feedback control systems, asymptotic stability requires that all closed-loop poles of the system transfer function lie strictly in the open left half of the complex s-plane. When a system is subjected to bounded disturbances, proportional-integral-derivative controllers compute corrective control actions based on instantaneous error, cumulative accumulated error, and error rate of change. Improper tuning of the derivative gain term amplifies high-frequency measurement noise, which can induce severe actuator oscillations and drive nonlinear systems into unstable limit-cycle trajectories.",
        "ai_text": "Closed-loop feedback control architectures regulate dynamic plant stability by continuously computing error offsets between reference setpoints and measured outputs. Proportional-integral-derivative algorithms calculate corrective inputs by integrating historical error offsets and differentiating instantaneous error trajectories. Excessive derivative gains amplify sensor noise, generating high-frequency actuator jitter and compromising system stability margins across complex industrial processes.",
        "human_source": "Feedback Systems: An Introduction / Åström & Murray",
        "human_url": "https://www.cds.caltech.edu/~murray/amwiki/",
        "generator": "GPT-4"
    },
    {
        "domain": "Technical / Engineering",
        "topic": "Hydraulic Pressure Distribution",
        "human_text": "Pascal's principle states that when pressure is applied to a confined, incompressible fluid, the pressure change is transmitted undiminished throughout the fluid in all directions and to the walls of the containing vessel. In hydraulic disc braking systems, applying a modest mechanical force to a small master cylinder piston generates hydraulic fluid pressure that acts over significantly larger slave caliper pistons. Because pressure equals force divided by area, the output clamping force exerted on the brake pads is magnified in direct proportion to the ratio of piston surface areas.",
        "ai_text": "Hydraulic power transmission systems exploit Pascal's principle to achieve mechanical force multiplication across enclosed fluid circuits. When compressive force is exerted on a primary master piston, pressure propagates uniformly through the hydraulic fluid matrix. Applying this pressurized fluid to secondary actuator pistons of larger cross-sectional area produces a proportionally amplified output clamping force, enabling powerful mechanical braking.",
        "human_source": "University Physics / OpenStax",
        "human_url": "https://openstax.org/details/books/university-physics-volume-1",
        "generator": "Claude 3.5 Sonnet"
    },

    # 2. Computer Science
    {
        "domain": "Computer Science",
        "topic": "CPU Cache Hierarchy & Memory Latency",
        "human_text": "Modern computer architectures bridge the vast speed disparity between processor registers and dynamic random-access memory by organizing storage into a hierarchical cache subsystem. Small, extremely fast static RAM caches (L1 and L2) are integrated directly onto the processor core to store recently referenced cache lines. When the CPU issues a memory read, the cache controller checks tag arrays for a hit; if a cache miss occurs, the execution pipeline stalls while the missing line is retrieved from larger, slower L3 cache or off-chip main memory.",
        "ai_text": "Multi-tier cache memory hierarchies mitigate the memory latency wall by exploiting temporal and spatial locality in instruction and data access streams. L1 and L2 SRAM caches operate at near-processor clock frequencies to deliver low-latency line lookups. When cache miss events occur, memory management controllers fetch contiguous line blocks from shared L3 cache or DRAM channels, managing hardware eviction protocols like LRU to sustain high compute throughput.",
        "human_source": "Computer Systems: A Programmer's Perspective / Bryant & O'Hallaron",
        "human_url": "https://csapp.cs.cmu.edu/",
        "generator": "Gemini 1.5 Pro"
    },
    {
        "domain": "Computer Science",
        "topic": "Garbage Collection Mark-Sweep Algorithm",
        "human_text": "The classic mark-and-sweep garbage collection algorithm operates in two sequential phases to reclaim unreachable heap objects in managed runtime environments. In the mark phase, the collector traverses the directed graph of active references starting from known root pointers—including CPU registers, execution stack frames, and global variables—setting a mark bit on every reachable object. In the subsequent sweep phase, the runtime scans the entire contiguous heap memory linearly, appending unmarked memory blocks to a free allocation list and clearing mark bits on surviving live objects.",
        "ai_text": "Mark-and-sweep garbage collectors automate heap memory management by identifying unreferenced data objects and returning memory pages to free allocation pools. The garbage collection runtime begins by traversing the root reference graph across thread execution stacks and global pointer tables, flagging visited objects as reachable. A subsequent linear sweep across heap regions reclaims unmarked memory allocations, mitigating fragmentation while resetting object header flags for future collection cycles.",
        "human_source": "Crafting Interpreters / Bob Nystrom",
        "human_url": "https://craftinginterpreters.com/",
        "generator": "LLaMA 3 70B"
    },

    # 3. Artificial Intelligence / Machine Learning
    {
        "domain": "Artificial Intelligence / Machine Learning",
        "topic": "Backpropagation Chain Rule",
        "human_text": "Backpropagation calculates the gradient of a scalar objective loss function with respect to all trainable network parameters by systematically applying the calculus chain rule backwards through computational graph nodes. During the forward pass, intermediate activation vectors and linear pre-activations are cached in memory. In the backward pass, upstream error gradients are multiplied by local Jacobian matrices, allowing the optimization algorithm to compute parameter weight updates across deep multilayer architectures.",
        "ai_text": "The backpropagation algorithm drives deep neural network parameter optimization by systematically propagating loss gradients backwards across computational graph layers. Utilizing the multivariable calculus chain rule, the algorithm multiplies incoming gradient tensors by localized activation derivative matrices. These computed gradients scale weight adjustments during gradient descent updates, minimizing empirical prediction error over training batches.",
        "human_source": "Deep Learning / Goodfellow, Bengio, Courville (MIT Press)",
        "human_url": "https://www.deeplearningbook.org/",
        "generator": "Mistral Large"
    },
    {
        "domain": "Artificial Intelligence / Machine Learning",
        "topic": "Transformer Multi-Head Self-Attention",
        "human_text": "In the standard Transformer architecture, multi-head self-attention allows token representations to simultaneously attend to contextual information from different representation subspaces at different sequence positions. Input vectors are linearly projected into Query, Key, and Value matrices. The scaled dot-product attention computes softmax-normalized compatibility scores between queries and keys, weighting the value vectors before projecting through a final output matrix.",
        "ai_text": "Multi-head self-attention mechanisms enable neural Transformer models to capture complex token dependencies across arbitrary sequence lengths. Linear projection layers map embedded token inputs into distinct Query, Key, and Value spaces. Scaled dot-product computations calculate attention score weights across sequence positions, aggregating contextual value vectors to yield contextualized embeddings.",
        "human_source": "The Illustrated Transformer / Jay Alammar",
        "human_url": "https://jalammar.github.io/illustrated-transformer/",
        "generator": "GPT-4"
    },

    # 4. Academic / Scientific
    {
        "domain": "Academic / Scientific",
        "topic": "CRISPR-Cas9 Gene Editing Specificity",
        "human_text": "The bacterial CRISPR-Cas9 adaptive immune system has been adapted into a powerful molecular tool for targeted genomic engineering. The Cas9 endonuclease is directed to specific DNA loci by a single-guide RNA molecule containing a twenty-nucleotide spacer sequence complementary to the target strand. Upon binding a protospacer adjacent motif (PAM) sequence (5'-NGG-3'), Cas9 unwinds the double helix and introduces a targeted double-strand break, triggering endogenous cellular repair pathways such as non-homologous end joining or homology-directed repair.",
        "ai_text": "CRISPR-Cas9 technologies facilitate site-specific genomic modifications by utilizing programmable single-guide RNA sequences to direct endonuclease cleavage. Upon recognizing adjacent protospacer motif sequences within target DNA segments, the Cas9 enzyme induces double-stranded breaks. Endogenous cellular repair mechanisms resolve these DNA breaks through non-homologous end joining or homology-directed repair pathways, enabling precise genetic knockouts or gene insertions.",
        "human_source": "Nature Reviews Genetics / Doudna & Charpentier",
        "human_url": "https://www.nature.com/nrg/",
        "generator": "Claude 3.5 Sonnet"
    },
    {
        "domain": "Academic / Scientific",
        "topic": "Photoelectric Effect and Quantum Hypothesis",
        "human_text": "Albert Einstein's 1905 explanation of the photoelectric effect established that electromagnetic radiation carries energy in discrete quantized packets termed photons. When light strikes a clean metallic surface, an electron is ejected only if the incident photon energy (E = hf) exceeds the metal's characteristic work function. Increasing light intensity increases the number of ejected photoelectrons per second but has no effect on their maximum kinetic energy, which depends strictly on the radiation frequency.",
        "ai_text": "The photoelectric effect demonstrates the particle nature of light by showing that electron emission from metallic surfaces depends on photon frequency rather than wave intensity. Individual photons transfer energy quantized by Planck's constant (E = hf) to conduction electrons. Photoemission occurs exclusively when photon energy surpasses the material's work function threshold, with surplus energy converted into electron kinetic energy.",
        "human_source": "University Physics with Modern Physics / OpenStax",
        "human_url": "https://openstax.org/details/books/university-physics-volume-3",
        "generator": "Gemini 1.5 Pro"
    },

    # 5. Educational
    {
        "domain": "Educational",
        "topic": "Mitochondria and Cellular Respiration",
        "human_text": "Mitochondria are membrane-bound cellular organelles often referred to as the powerhouses of eukaryotic cells because they generate the bulk of cellular adenosine triphosphate (ATP). Cellular respiration begins with glycolysis in the cytoplasm, converting glucose into pyruvate. Pyruvate enters the mitochondrial matrix, where the citric acid cycle generates high-energy electron carriers (NADH and FADH2). These molecules donate electrons to the inner mitochondrial membrane's electron transport chain, creating a proton gradient that drives ATP synthase.",
        "ai_text": "Mitochondria serve as vital energy-producing organelles in eukaryotic organisms by synthesizing adenosine triphosphate through cellular respiration. Glucose molecules undergo initial glycolysis in the cytoplasm before entering mitochondrial pathways, including the Krebs cycle and oxidative phosphorylation. Electron transport chains establish electrochemical proton gradients across the inner membrane, fueling ATP synthase to power essential cellular functions.",
        "human_source": "Concepts of Biology / OpenStax",
        "human_url": "https://openstax.org/details/books/concepts-biology",
        "generator": "LLaMA 3 70B"
    },
    {
        "domain": "Educational",
        "topic": "Plate Tectonics and Continental Drift",
        "human_text": "The theory of plate tectonics describes how Earth's outer rigid shell, the lithosphere, is broken into large structural plates that float on the ductile asthenosphere below. Thermal convection currents within Earth's mantle drive plate movement at rates of several centimeters per year. Where tectonic plates diverge along mid-ocean ridges, new crust is formed. Where plates collide at convergent boundaries, one plate is subducted beneath the other, generating deep ocean trenches, volcanic arcs, and mountain ranges.",
        "ai_text": "Plate tectonics explains the dynamic geological evolution of Earth's crust as lithospheric plates interact across the ductile upper mantle. Convective heat circulation in the mantle propels continental and oceanic plates across divergent, convergent, and transform boundaries. These boundary interactions drive seafloor spreading, seismic activity, volcanic eruptions, and orogenic mountain formation over geological timescales.",
        "human_source": "Physical Geology / OpenStax",
        "human_url": "https://openstax.org/details/books/physical-geology",
        "generator": "Mistral Large"
    },

    # 6. Social Media
    {
        "domain": "Social Media",
        "topic": "Developer Career Advice",
        "human_text": "Spent 6 hours debugging a distributed race condition today only to realize I had a typo in a redis key prefix. Reminder that no matter how senior you get, 90% of engineering is just patience, good logging, and taking a walk when you're stuck. Don't let imposter syndrome win.",
        "ai_text": "Spent the afternoon diving deep into a complex distributed caching issue, only to discover a subtle typo in a key prefix. A great reminder that no matter your experience level, debugging requires patience, robust logging, and occasional breaks. Keep pushing forward and trust the process! 💻🚀",
        "human_source": "Tech Social Media / Public Developer Posts",
        "human_url": "https://x.com",
        "generator": "GPT-4"
    },
    {
        "domain": "Social Media",
        "topic": "Product Launch Announcement",
        "human_text": "After 8 months of quiet building, we just pushed our new open-source vector database to github! Zero dependencies, written in pure Go, with built-in HNSW indexing. Would love your feedback and stars: link in bio.",
        "ai_text": "Excited to officially launch our new open-source vector search engine on GitHub! Built with zero external dependencies in Go, it delivers blazing-fast HNSW indexing and seamless developer integration. Check out the link in bio and let us know your thoughts! ⭐🔥",
        "human_source": "Open Source Community Post",
        "human_url": "https://github.com",
        "generator": "Claude 3.5 Sonnet"
    },

    # 7. Marketing / Advertising
    {
        "domain": "Marketing / Advertising",
        "topic": "Enterprise Cloud Migration Platform",
        "human_text": "Migrating mission-critical infrastructure to the cloud shouldn't mean weeks of downtime and hidden consulting costs. CloudBridge automates schema translation, pipeline verification, and real-time database replication with sub-millisecond lag. Schedule an architectural review with our engineering team today.",
        "ai_text": "Unlock seamless digital transformation with CloudBridge, the next-generation enterprise cloud migration platform. Effortlessly accelerate your infrastructure modernization with automated schema translation, real-time data replication, and enterprise-grade security. Empower your team to scale without limits today.",
        "human_source": "Enterprise SaaS Marketing Copy",
        "human_url": "https://example.com/saas",
        "generator": "Gemini 1.5 Pro"
    },
    {
        "domain": "Marketing / Advertising",
        "topic": "Ultra-Lightweight Ergonomic Running Shoes",
        "human_text": "Engineered with nitrogen-infused foam and a breathable carbon-plate weave, the AeroStride 4 weighs under 180 grams while delivering 85% energy return on every stride. Available now in select flagship stores and online.",
        "ai_text": "Experience the pinnacle of athletic performance with AeroStride 4. Featuring revolutionary nitrogen-infused cushioning and dynamic carbon-fiber plate technology, this ultra-lightweight running shoe elevates your training with unmatched energy return and comfort. Step into the future of running today.",
        "human_source": "Product Launch Press Release",
        "human_url": "https://example.com/gear",
        "generator": "LLaMA 3 70B"
    },

    # 8. Real Estate
    {
        "domain": "Real Estate",
        "topic": "Modern Luxury Penthouse Listing",
        "human_text": "Rarely available top-floor corner penthouse offering 3,200 square feet of sun-drenched living space with panoramic skyline views. Features include wide-plank white oak flooring, a custom chef's kitchen with Sub-Zero appliances, private elevator entry, and an expansive 800 sq ft wrap-around terrace. Steps from the waterfront park and transit.",
        "ai_text": "Welcome to your dream urban sanctuary in the heart of the city. This breathtaking luxury penthouse boasts soaring ceilings, expansive floor-to-ceiling windows, and panoramic skyline vistas. Featuring an immaculate gourmet kitchen, bespoke finishes, and a private wraparound terrace, this property represents the epitome of refined luxury living.",
        "human_source": "Verified MLS Listing Copy",
        "human_url": "https://mls.com",
        "generator": "Mistral Large"
    },
    {
        "domain": "Real Estate",
        "topic": "Suburban Craftsman Family Home",
        "human_text": "Charming 4-bedroom Craftsman home in the sought-after Westwood school district. Highlights include a renovated open-concept kitchen with quartz counters, original stone fireplace, finished basement with home theater, and a fenced backyard with mature cedar trees. Open house this Saturday 1-4pm.",
        "ai_text": "Discover timeless elegance and contemporary comfort in this stunning four-bedroom Craftsman residence located in an exceptional neighborhood. Boasting a beautifully renovated gourmet kitchen, sunlit living spaces, and a picturesque private backyard, this home offers the perfect blend of family living and entertaining.",
        "human_source": "Real Estate Brokerage Sheet",
        "human_url": "https://realtor.com",
        "generator": "GPT-4"
    },

    # 9. Blogs / Articles
    {
        "domain": "Blogs / Articles",
        "topic": "Remote Work and Team Communication",
        "human_text": "The hardest part about managing distributed teams isn't tracking Jira tickets or setting up Zoom links—it's building enough psychological safety that engineers speak up early when a project is slipping. When daily hallway chatter disappears, managers have to replace casual observation with intentional asynchronous check-ins.",
        "ai_text": "Navigating remote team leadership requires intentional communication strategies and a deep commitment to psychological safety. In asynchronous work environments, successful engineering leaders replace passive monitoring with structured check-ins, transparent documentation, and continuous team collaboration to foster long-term project success.",
        "human_source": "Tech Leadership Blog",
        "human_url": "https://medium.com",
        "generator": "Claude 3.5 Sonnet"
    },
    {
        "domain": "Blogs / Articles",
        "topic": "The Psychology of Habit Formation",
        "human_text": "If you want to build a lasting habit, focus on the friction in your environment rather than relying on willpower. Putting your running shoes next to your bed or leaving a book on your pillow changes the cue-routine-reward loop by eliminating the initial resistance before you start.",
        "ai_text": "Cultivating sustainable habits is fundamentally about optimizing your daily environment rather than relying solely on motivation. By reducing friction around desired behaviors and restructuring cognitive cues, individuals can seamlessly integrate positive routines into their lifestyle and achieve meaningful personal growth.",
        "human_source": "Personal Development Essay",
        "human_url": "https://substack.com",
        "generator": "Gemini 1.5 Pro"
    },

    # 10. News / Journalism
    {
        "domain": "News / Journalism",
        "topic": "Central Bank Interest Rate Decision",
        "human_text": "The Federal Reserve held benchmark interest rates steady at 5.25% to 5.50% on Wednesday, citing resilient labor markets and persistent service-sector inflation. In his post-meeting press conference, the chair emphasized that policymakers require additional convincing evidence that inflation is moving sustainably toward the 2% target before initiating rate cuts.",
        "ai_text": "Federal Reserve officials voted unanimously to maintain benchmark interest rates unchanged on Wednesday, underscoring ongoing progress against inflation alongside continued economic resilience. Central bank policymakers reaffirmed their data-dependent approach, signaling that interest rate reductions will remain contingent on sustained disinflationary trends.",
        "human_source": "Reuters Financial News",
        "human_url": "https://reuters.com",
        "generator": "LLaMA 3 70B"
    },
    {
        "domain": "News / Journalism",
        "topic": "Commercial Satellite Constellation Launch",
        "human_text": "A commercial rocket lifted off from Cape Canaveral Space Force Station early Thursday morning, deploying 24 broadband communications satellites into low Earth orbit. The booster successfully touched down on a drone ship in the Atlantic Ocean approximately eight minutes after liftoff, marking its twelfth completed flight.",
        "ai_text": "A heavy-lift orbital rocket successfully launched from Cape Canaveral on Thursday, deploying the latest batch of next-generation communications satellites into orbit. Following first-stage stage separation, the reusable booster executed a precision landing on an autonomous drone ship, highlighting continued advances in commercial spaceflight efficiency.",
        "human_source": "Associated Press Wire",
        "human_url": "https://apnews.com",
        "generator": "Mistral Large"
    },

    # 11. Conversational / Q&A
    {
        "domain": "Conversational / Q&A",
        "topic": "Explaining Why the Sky is Blue",
        "human_text": "The sky looks blue because sunlight gets scattered by nitrogen and oxygen molecules in our atmosphere. Blue light travels in shorter, smaller waves than red light, so it gets scattered in every direction much more easily. When you look up away from the sun, you see all that scattered blue light.",
        "ai_text": "The blue appearance of the daytime sky is caused by a physical phenomenon known as Rayleigh scattering. Sunlight consists of a spectrum of colors with different wavelengths. Because shorter blue wavelengths scatter much more efficiently off atmospheric gas particles than longer red wavelengths, blue light fills the sky in all directions.",
        "human_source": "HC3 Human Explanations",
        "human_url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3",
        "generator": "GPT-4"
    },
    {
        "domain": "Conversational / Q&A",
        "topic": "Difference Between HTTP and HTTPS",
        "human_text": "HTTP sends all your website data in plain clear text, meaning anyone intercepting your Wi-Fi can see passwords or credit card numbers you type. HTTPS wraps that exact same web traffic inside an encrypted TLS connection, so only you and the web server can read the transmitted data.",
        "ai_text": "The primary difference between HTTP and HTTPS is encryption and security. While standard HTTP transmits data in unencrypted plaintext, HTTPS utilizes Transport Layer Security (TLS) certificates to encrypt communication channels between clients and servers, preventing eavesdropping and tampering.",
        "human_source": "StackExchange Q&A Community",
        "human_url": "https://superuser.com",
        "generator": "Claude 3.5 Sonnet"
    },

    # 12. Product Reviews
    {
        "domain": "Product Reviews",
        "topic": "Noise-Cancelling Wireless Headphones",
        "human_text": "After two weeks of daily commuting on the subway, the active noise cancellation on these headphones is top-notch. Battery life easily lasts 30 hours on a single charge. The only minor gripe is the ear cups get a bit warm during long listening sessions, but the soundstage is crisp and punchy.",
        "ai_text": "These wireless noise-canceling headphones deliver exceptional audio performance and class-leading noise cancellation across diverse environments. With an impressive 30-hour battery life, intuitive controls, and a balanced acoustic profile, they represent an outstanding investment for audiophiles and daily commuters alike.",
        "human_source": "Verified Hardware Review / Forum Post",
        "human_url": "https://head-fi.org",
        "generator": "Gemini 1.5 Pro"
    },
    {
        "domain": "Product Reviews",
        "topic": "Mechanical Espresso Coffee Machine",
        "human_text": "Solid build quality with heavy stainless steel housing. The dual boiler heats up in under 3 minutes and the steam wand has plenty of power for microfoam. Steaming milk while pulling a double shot with consistent 9-bar pressure makes morning routines much faster.",
        "ai_text": "This premium espresso machine combines commercial-grade engineering with user-friendly operation. Featuring dual boiler technology, precise PID temperature control, and a high-powered steam wand, it consistently delivers rich, barista-quality espresso and velvety microfoam right in your home kitchen.",
        "human_source": "Coffee Enthusiast Forum",
        "human_url": "https://home-barista.com",
        "generator": "LLaMA 3 70B"
    },

    # 13. Finance
    {
        "domain": "Finance",
        "topic": "Corporate Bond Yield and Duration Risk",
        "human_text": "Bond duration measures a fixed-income portfolio's sensitivity to interest rate fluctuations. When market interest rates rise, the present value of future coupon payments and principal repayment falls, causing bond prices to decline. For investment-grade corporate bonds with long maturities, higher modified duration exposes investors to greater capital losses during rate hike cycles.",
        "ai_text": "Bond duration serves as a foundational risk metric that quantifies the price volatility of fixed-income instruments in response to interest rate shifts. As benchmark rates increase, bond prices fall inversely to align yield curves with prevailing market conditions. Portfolio managers adjust duration exposures to manage interest rate risk across corporate debt holdings.",
        "human_source": "Corporate Finance Principles / CFA Institute",
        "human_url": "https://cfainstitute.org",
        "generator": "Mistral Large"
    },
    {
        "domain": "Finance",
        "topic": "Venture Capital Term Sheet Liquidation Preferences",
        "human_text": "A 1x non-participating liquidation preference ensures that in a sale or liquidity event, preferred shareholders receive their original invested capital back before common stockholders receive proceeds. If the exit valuation is high enough that converting preferred shares to common stock yields a higher return, investors waive the preference and share pro rata.",
        "ai_text": "Liquidation preference provisions in venture capital term sheets dictate the distribution order of financial proceeds during corporate exit or acquisition events. Preferred equity holders typically secure priority liquidation claims, ensuring initial investment recovery before remaining assets are allocated to common equity holders and founders.",
        "human_source": "Venture Deals / Brad Feld & Jason Mendelson",
        "human_url": "https://feld.com",
        "generator": "GPT-4"
    },

    # 14. Healthcare / Medical
    {
        "domain": "Healthcare / Medical",
        "topic": "Type 2 Diabetes Pathophysiology",
        "human_text": "Type 2 diabetes mellitus is characterized by progressive peripheral insulin resistance coupled with relative beta-cell secretory dysfunction in the pancreatic islets. When skeletal muscle, adipose, and hepatic tissues fail to respond adequately to circulating insulin, cellular glucose uptake decreases and hepatic gluconeogenesis remains unsuppressed, resulting in persistent hyperglycemia.",
        "ai_text": "Type 2 diabetes is a chronic metabolic condition characterized by systemic insulin resistance and impaired insulin secretion from pancreatic beta cells. Ineffective glucose uptake in peripheral tissues leads to elevated blood glucose levels, prompting compensatory hyperinsulinemia that can eventually progress to beta-cell exhaustion and diabetic vascular complications.",
        "human_source": "Harrison's Principles of Internal Medicine / McGraw-Hill",
        "human_url": "https://accessmedicine.mhmedical.com",
        "generator": "Claude 3.5 Sonnet"
    },
    {
        "domain": "Healthcare / Medical",
        "topic": "Cardiac Conduction System & Arrhythmias",
        "human_text": "Electrical activation in the heart initiates spontaneously in the sinoatrial node, spreading through the atria to the atrioventricular node. The AV node introduces an intrinsic delay that allows atrial contraction to complete ventricular filling before rapid conduction occurs down the bundle of His and Purkinje fiber network, coordinating synchronous ventricular systole.",
        "ai_text": "The cardiac conduction system coordinates rhythmic myocardial contractions through organized electrical impulses originating in the sinoatrial node. The atrioventricular node regulates impulse transmission speed, allowing complete atrial emptying into ventricular chambers before activating Purkinje networks to initiate synchronized ventricular ejection.",
        "human_source": "Anatomy & Physiology / OpenStax",
        "human_url": "https://openstax.org/details/books/anatomy-and-physiology-2e",
        "generator": "Gemini 1.5 Pro"
    },

    # 15. Legal / Formal
    {
        "domain": "Legal / Formal",
        "topic": "Indemnification and Limitation of Liability",
        "human_text": "Each party agrees to defend, indemnify, and hold harmless the other party and its respective officers, directors, and employees from and against any third-party claims, liabilities, damages, or costs arising out of any material breach of the confidentiality obligations or gross negligence under this Agreement.",
        "ai_text": "The Service Provider shall indemnify, defend, and hold harmless the Client against any third-party claims, liabilities, losses, or legal expenses resulting from a material breach of warranty, intellectual property infringement, or willful misconduct in connection with the performance of Services under this Agreement.",
        "human_source": "SEC EDGAR Public Commercial Filing",
        "human_url": "https://sec.gov/edgar",
        "generator": "LLaMA 3 70B"
    },
    {
        "domain": "Legal / Formal",
        "topic": "Force Majeure Clause",
        "human_text": "Neither party shall be liable for failure or delay in performing its contractual obligations if such failure arises from causes beyond its reasonable control, including acts of God, war, riot, civil commotion, labor disputes, failure of electrical or telecommunications infrastructure, or government embargoes.",
        "ai_text": "Neither party shall be deemed in breach of its obligations under this Agreement to the extent that performance is prevented or delayed by events of Force Majeure, including natural disasters, acts of war, civil unrest, labor strikes, power grid failures, or governmental restrictions beyond reasonable control.",
        "human_source": "Commercial Contract Standards / American Bar Association",
        "human_url": "https://americanbar.org",
        "generator": "Mistral Large"
    },

    # 16. Creative Writing / Stories
    {
        "domain": "Creative Writing / Stories",
        "topic": "Storm Approaching a Coastal Lighthouse",
        "human_text": "The wind picked up an hour before dusk, driving salt spray against the thick lantern glass of the lighthouse. Thomas adjusted the counterweights, listening to the heavy iron gears click in their rhythm as the revolving reflector swept its yellow beam into the gathering black fog over the reef.",
        "ai_text": "The tempest descended upon the jagged coastline as dusk faded into midnight. Inside the solitary lighthouse tower, the veteran keeper tended the rotating brass beacon, watching the luminous light cut through the billowing fog as waves crashed relentlessly against the stone foundation below.",
        "human_source": "Short Story Fiction Archive",
        "human_url": "https://gutenberg.org",
        "generator": "GPT-4"
    },
    {
        "domain": "Creative Writing / Stories",
        "topic": "Distant Space Exploration Arrival",
        "human_text": "After three centuries in cryogenic drift, the colony vessel shuddered as attitude thrusters fired their deceleration burn. On the bridge observation screens, the amber rings of the gas giant resolved from blurred pinpricks of light into massive bands of swirling ice crystals.",
        "ai_text": "Emerging from centuries of deep space transit, the exploration vessel glided toward the uncharted exoplanet. As auxiliary thrusters stabilized the ship's orbital trajectory, the planetary observation monitors illuminated with vibrant sapphire oceans and sprawling continent formations.",
        "human_source": "Speculative Fiction Collection",
        "human_url": "https://gutenberg.org",
        "generator": "Claude 3.5 Sonnet"
    },

    # 17. Tutorials / How-To
    {
        "domain": "Tutorials / How-To",
        "topic": "Setting Up an SSH Key Pair",
        "human_text": "To generate a modern SSH key pair, open your terminal and run `ssh-keygen -t ed25519 -C 'your_email@example.com'`. When prompted, press Enter to accept the default file location and provide a strong passphrase. Next, copy your public key to your remote server using `ssh-copy-id username@remote_host`.",
        "ai_text": "To configure secure SSH key authentication, begin by executing `ssh-keygen -t ed25519 -C 'your_email@example.com'` in your command terminal. Save the key to the default directory and specify an optional passphrase. Finally, transfer your public key to the remote destination host with `ssh-copy-id user@host`.",
        "human_source": "OpenSSH Documentation / Ubuntu Server Guide",
        "human_url": "https://ubuntu.com/server/docs",
        "generator": "Gemini 1.5 Pro"
    },
    {
        "domain": "Tutorials / How-To",
        "topic": "Configuring a Reverse Proxy with NGINX",
        "human_text": "To set up NGINX as a reverse proxy for a local Node.js application, add a `location /` block inside your server block containing `proxy_pass http://127.0.0.1:3000;`. Make sure to pass along headers such as `proxy_set_header Host $host;` and `proxy_set_header X-Real-IP $remote_addr;` before running `nginx -t` and reloading the service.",
        "ai_text": "Configuring NGINX as a reverse proxy forwards incoming HTTP client traffic to backend application servers. Within your site configuration file, define a `location /` directive with `proxy_pass http://localhost:3000;`. Include proxy headers for Host and client IP forwarding, test configuration syntax with `nginx -t`, and reload NGINX.",
        "human_source": "NGINX Community Documentation",
        "human_url": "https://nginx.org/en/docs/",
        "generator": "LLaMA 3 70B"
    },

    # 18. General Informational Writing
    {
        "domain": "General Informational Writing",
        "topic": "How Optical Fiber Transmits Data",
        "human_text": "Fiber optic cables transmit digital data by sending pulses of infrared or visible light through thin strands of ultra-pure silica glass. Light entering the core at an angle shallower than the critical angle undergoes total internal reflection, bouncing repeatedly off the boundary between the inner core and outer cladding layer without escaping.",
        "ai_text": "Optical fiber communication transmits information across long distances using modulated pulses of light guided through flexible glass cores. Utilizing the principle of total internal reflection, refractive index differences between the inner core and outer cladding trap light rays within the strand, delivering high-bandwidth transmission.",
        "human_source": "Encyclopaedia Britannica / Physics Articles",
        "human_url": "https://britannica.com",
        "generator": "Mistral Large"
    },
    {
        "domain": "General Informational Writing",
        "topic": "The Water Cycle and Evapotranspiration",
        "human_text": "The hydrologic cycle describes the continuous movement of water on, above, and below Earth's surface. Solar radiation evaporates liquid water from oceans and lakes into atmospheric vapor, while plants release additional moisture through transpiration in their leaves. As warm humid air rises and cools, water vapor condenses into clouds and eventually falls back to the surface as precipitation.",
        "ai_text": "The global water cycle represents the continuous circulation of moisture across Earth's hydrosphere, atmosphere, and lithosphere. Solar heat energy drives surface evaporation and plant evapotranspiration into the atmosphere. As rising moisture cools, condensation forms clouds that release freshwater back to terrestrial and marine ecosystems through precipitation.",
        "human_source": "USGS Water Science School",
        "human_url": "https://www.usgs.gov/special-topics/water-science-school",
        "generator": "GPT-4"
    }
]
