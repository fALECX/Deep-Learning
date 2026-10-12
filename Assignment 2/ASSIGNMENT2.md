# Assignment 2: Arbeitsdokument

> **ABGABE: Mittwoch, 14. Oktober 2026, 23:59.**
> Interne Deadline: **14.10., 18:00**, mit fertig hochgeladenem Notebook und PDF.
> Abzug bei Verspätung: 10 % pro Tag. PDF und Notebook müssen identisch sein, sonst gibt es 0 Punkte.

Abgabe: `A2_s4271512.ipynb` plus PDF-Export desselben Notebooks über den Turnitin-Link.
Spec: `Assignment 2_ Instructions and Notebook Submission.pdf`. Rubrik: `rubric-assignment2_2026.xlsx`.

---

## 1. Fakten, die den Plan bestimmen

| Befund | Zahl | Konsequenz |
|---|---|---|
| Split train/dev/test | 688 / 63 / 200 | Dev zu klein → Validierung aus train+dev neu stratifiziert ziehen, Test erst ganz am Ende |
| Imbalance | Loaded Language 489 … Bandwagon/Red Herring/Obfuscation je 5 | gewichteter Loss, Macro-F1 mitberichten, seltene Klassen **nicht** retten wollen |
| Labels/Meme | Ø 1.73, 199 ohne Label, max 8 | Multi-Label: Sigmoid + BCE |
| Textlänge | Median 15 Wörter, max 72 | max_len 128 Tokens reicht → schnell auf T4 |
| Span-Überlappung | 1394 überlappende Paare | kein BIO; IO-Tagging mit 20 Sigmoids pro Token |
| Label ⇔ Span | 100 % konsistent | Multi-Task: 1 Encoder, 2 selbst gebaute Köpfe |
| Groß-Schreibung | 468/951 Memes fast nur CAPS | Casing-Experiment (billig), wichtig für C4 |
| **Leakage** | ID `474_batch_2` steht in train *und* test, mit unterschiedlichen Labels | aus Training entfernen, dokumentieren |
| **Label-Rauschen** | gleicher Text, verschiedene Labels (z. B. "the real virus threatening america") | Obergrenze für erreichbares F1; im Judgment erwähnen |
| Politische Seite (Keyword-Heuristik, Test) | gegen links 33, gegen rechts 26, beide 10, keins 131 | Gruppenvergleich in C1 hat kaum Power → **kontrafaktischer Tausch** als Hauptexperiment |
| Paper | FB-Gruppen USA/Kanada, 2020, Englisch; 6 Annotatoren, Konsens; α = 0.83 / 0.91 / **0.56** | Basis für C1 und C4 |
| Referenz SemEval | A: Micro-F1 .593 (Best), .374 (Majority) · B: Span-F1 .482 (Best) | Maßstab |

## 2. Experimente mit Vorab-Hypothesen

Wir kürzen den Umfang **nicht** wegen der Deadline. Zu jedem Experiment schreiben wir vorher eine Hypothese und die Entscheidung, die daraus folgt. So ist jedes Experiment zielgerichtet, und keines läuft ins Leere.

| # | Experiment | Hypothese | Entscheidung daraus |
|---|---|---|---|
| E0 | Baselines: Majority, Prior-Random, Keyword nur zur Analyse | Majority ≈ .37 micro / .03 macro | Untergrenze |
| E1 | Encoder: RoBERTa-base vs. DeBERTa-v3-base vs. BERTweet-base (Social-Media-Domäne) | BERTweet/DeBERTa leicht vorn; CAPS-Text ist für alle schwierig | bester Encoder mit Evidenz |
| E2 | Adaption: frozen + Head vs. LoRA vs. Full Fine-Tune (Rubrik HD) | frozen ≈ .40–.45; LoRA ≈ Full ± Seed-Rauschen | bei Gleichstand das kleinere/schnellere Modell |
| E3 | Architektur: Joint Multi-Task vs. getrennte Modelle vs. "Label = OR der Spans" | Joint ≥ getrennt bei 688 Beispielen | Begründung für Part-B-Design |
| E4 | Label-Head: CLS vs. Mean- vs. Attention-Pooling (selbst gebautes Segment) | Attention ≥ CLS | Design des eigenen Segments |
| E5 | Loss: BCE vs. BCE+pos_weight vs. Focal/ASL | gewichtet hilft Macro-F1, kostet etwas Micro | Wahl nach Ziel-Metrik |
| E6 | HP-Suche (LR, Epochen, Dropout, Batch) mit 5-fold CV × Seeds | LR wichtigster Faktor | finale HPs |
| E7 | Schwellen: global vs. per-Class | per-Class overfittet bei wenig Positiven | Regel nach Positivenzahl |
| E8 | Casing: Original vs. normalisiert vs. Augmentation (zufälliges Lower/Upper) | Augmentation macht robuster gegen Schüler-Feeds | fließt in C4 ein |
| E9 | Zusatzdaten PTC (SemEval-2020 T11, Daten, kein Checkpoint), deklariert | kleiner Gewinn bei Overlap-Techniken, Domänen-Shift | nur übernehmen, wenn CV-Gewinn > Rauschen |
| E10 | Explainability: Integrated Gradients / Gradient × Input | Attribution trifft Loaded Words, deckt sich teils mit Span-Head | Interpretation + C3 |
| C1 | Gruppenvergleich (mit CIs) + **kontrafaktischer Tausch** (Trump↔Biden, Dem↔Rep, liberal↔conservative) + Label-Verteilung je Seite in den Annotationen | Flip-Rate klein, aber messbar; Gruppen-CIs breit | Fairness-Urteil mit Grenzen |
| C2 | Zero-Shot-LLM auf stratifizierter Stichprobe (~100), Temp 0 + Wiederholungen, Parser für falsches Format, Memorization-Test | Micro ≈ .30–.45, viele False Positives; Kontamination plausibel | Verteidigung des eigenen Systems |
| C3 | Disagreement Label- vs. Span-Head pro Technik | hoch bei abstrakten Techniken | UI-Regel |
| C4 | US→AU Entity-Swap auf Test (Labels bleiben gültig) + Stil-Perturbationen + **gelabeltes AU-2026-Probe-Set (~40, von mir geschrieben und gelabelt, als synthetisch deklariert)** + Tokenizer-Fragmentierung von AU-Slang | F1 fällt, Slang zerfällt in Subwörter | Deployment-Urteil + Lehrkraft-Workflow |

**Vorab erwartetes Endurteil:** Das System taugt als Diskussionshilfe *mit* Lehrkraft für die häufigen Techniken. Es ist nicht autonom einsetzbar und nicht für seltene Techniken. Bei AU-Inhalten sollte es nur mit Monitoring laufen.

## 3. Arbeitsweise und Compute

- **Lokal (automatisiert, ich):** Code entwickeln, Unit-Checks, kleine Experimente im Hintergrund auf der CPU (RoBERTa-base ≈ 1,5 s/Step → ca. 12 min pro Lauf). Die Intel-Arc-GPU (XPU) stürzt beim Training ab (Segfault/OOM), sie ist daher nicht nutzbar.
- **GPU-Grid (automatisiert, wenn eingerichtet):** Kaggle-Kernels per CLI (kostenlose T4, ~30 h/Woche). Ich pushe, starte und lade Ergebnisse runter.
- **Abschluss (du):** Restart & Run All auf Colab T4 (Pflicht laut Spec), PDF-Export, Upload.
- **Nicht** der EMR-Cluster: Er gehört zum Big-Data-Kurs COSC2637 und hat keine GPU, nur Spark/Hadoop.

## 3b. Checkliste

- [x] Daten, Paper, Rubrik analysiert, Plan steht
- [ ] Notebook-Gerüst: Setup, Seed, EDA mit Plots, Leakage-Fix
- [ ] Eval-Framework: CV-Splits, Metriken A+B (SemEval-Span-F1 gegen offiziellen Scorer geprüft), Baselines (E0)
- [ ] Modellcode Multi-Task (lokaler Mini-Lauf)
- [ ] C2-Runner (LLM) bauen und laufen lassen, früh, weil blockierend
- [ ] E1–E9 (Grid)
- [ ] Finales Modell, Test-Evaluation mit Bootstrap-CI, E10
- [ ] C1, C3, C4
- [ ] Ultimate Judgment, Limitations
- [ ] Restart & Run All auf Colab, PDF-Export, Upload

## 4. Entscheidungen

- 2026-10-12: C2 nutzt ein offenes Modell auf Colab (Qwen2.5-7B-Instruct 4-bit, Temperatur 0, + 2 Läufe mit Temp 0.7 für Variabilität).
- 2026-10-12: **C2-Gegner:** (1) Gemini über grsai (`api.grsai.com`, Gemini-native `/v1beta/models/{model}:generateContent`, Key in `~/.env` als `GRSAI_API_KEY`). Funktionierende IDs: `gemini-3.1-flash-lite`, `gemini-3.1-pro`. Hinweis für die Limitations: grsai meldet für einen 2-Wort-Prompt ca. 87–91 Prompt-Tokens → vermutlich injizierter Wrapper/System-Prompt; die Version ist nur über `modelVersion` im Response belegt. (2) Lokal: Ornith-1.5-9B Q4_K_M (Qwen3.5-Basis, MIT, Reasoning-Modell) über LM Studio = die "auf T4 einsetzbare" Alternative.
- 2026-10-12: Daten werden ins private Repo gepusht (User-Entscheidung). API-Keys nie ins Repo.
- 2026-10-12: Lokale Pakete installiert (transformers 5.19, peft 0.21, pandas 3.0). Training läuft auf Colab.
- 2026-10-12: Umfang wird nicht wegen der Deadline gekürzt (Wunsch des Users). C4 = Entity-Swap + Perturbation + selbst geschriebenes, gelabeltes Probe-Set.

## 5. Lernliste: was du verstehen musst (und wo im Kurs)

| Thema | Wofür | Kurs |
|---|---|---|
| Multi-Label vs. Multi-Class (Sigmoid+BCE vs. Softmax+CE) | Part A | W2–W3 |
| Micro- vs. Macro-F1, Schwellenwahl | Evaluation | W6 |
| Train/Val/Test-Trennung, Leakage | Rubrik: "leaky eval" = 0 | W6 |
| Tokenizer, Subwörter, `offset_mapping` (Zeichen → Token) | Span-Labels erzeugen | W8 |
| Token-Klassifikation (NER), BIO vs. IO | Part B | W7–W8 |
| Fine-Tuning vs. Frozen vs. LoRA | Rubrik HD | W9 |
| Class Imbalance (pos_weight) | seltene Techniken | W6 |
| Seeds, Varianz, Bootstrap-Konfidenzintervalle | ehrliche Zahlen | W6 |
| Gradient-Attribution / Integrated Gradients | Explainability | W11–W12 |
| Kontrafaktische Tests (Fairness) | C1 | Literatur |
| Datenkontamination bei LLMs | C2 | Literatur |
