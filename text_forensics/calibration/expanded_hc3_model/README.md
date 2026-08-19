# Expanded HC3 Forensic Model (Experimental V2 Candidate)

- **Architecture**: StandardScaler + LogisticRegression (5 features)
- **Trained Samples**: 508
- **Thresholds**: {"human_max": 0.36, "likely_human_max": 0.55, "likely_ai_min": 0.55, "ai_min": 0.75}
- **Coefficients**: {"curvature": 2.14946617996355, "burstiness": -0.8876702907518279, "lexical_entropy": -1.284525607847812, "structural_regularity": 0.06240250961872482, "cliche_density": 1.4767997438993612}
- **Locked Test F1**: 92.93%
