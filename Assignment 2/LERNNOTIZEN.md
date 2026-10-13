# Lernnotizen zu Assignment 2

Zu jeder Designentscheidung im Notebook steht hier: **was** wir machen, **warum**, und **was du dazu sagen können musst**. Die Abschnitte folgen dem Notebook.

---

## 1. Daten verstehen (Notebook §1)

**Multi-Label statt Multi-Class.** Ein Meme kann 0, 1 oder bis zu 8 Techniken haben.
- Bei *Multi-Class* gibt es Softmax → genau eine Klasse, und alle Wahrscheinlichkeiten summieren sich zu 1.
- Bei *Multi-Label* bekommt jede Technik ein eigenes Sigmoid und eine eigene Ja/Nein-Entscheidung → 20 unabhängige binäre Klassifikatoren auf einer gemeinsamen Repräsentation.
- Loss: Binary Cross-Entropy pro Technik (`BCEWithLogitsLoss`).

**Long Tail.** Loaded Language kommt 489-mal vor, Red Herring 5-mal (davon 1× im Training). Kein Verfahren lernt etwas aus einem einzigen Beispiel.
- *Merksatz:* "Wir versprechen nicht, seltene Techniken zu erkennen. Wir messen und sagen ehrlich, welche das Tool unterstützt."

**Leakage.** Meme `474_batch_2` steht in Train **und** Test.
- Ohne Fix würde das Modell eine Testantwort "auswendig" kennen.
- Wir entfernen es aus dem Training und lassen den Test unverändert. So bleiben wir mit den offiziellen Ergebnissen vergleichbar.

**Label-Rauschen.** Gleicher Text, unterschiedliche Labels.
- Laut Paper lag die Übereinstimmung der Annotatoren bei α = 0.56–0.91.
- *Folge:* Es gibt eine Obergrenze. Kein Modell kann konsistenter sein als die Menschen, deren Labels es lernt.

**Provenienz.** Die Daten kommen aus US/kanadischen Facebook-Gruppen von 2020, gelabelt von 6 Annotatoren mit Konsens-Diskussion.
- Darauf bauen C1 und C4 auf: "korrekt" heißt "stimmt mit *diesem* Team überein".

## 2. Evaluation (§2)

**Micro- vs. Macro-F1.**
- *Micro:* Man zählt alle TP/FP/FN über alle Techniken zusammen und rechnet daraus ein F1. Häufige Klassen dominieren.
- *Macro:* Man berechnet F1 pro Technik und mittelt dann. Jede Technik zählt gleich viel.
- Wir berichten beides: Micro wegen der Vergleichbarkeit mit SemEval, Macro wegen des Unterrichts.

**Span-F1 mit Teilpunkten.** Für einen vorhergesagten Span p und einen Gold-Span g mit gleicher Technik:
- Precision = |p ∩ g| / |p|, Recall = |p ∩ g| / |g|, gerechnet in Zeichen.
- Wir haben das selbst implementiert und gegen das **offizielle Scorer-Skript** geprüft: identisch.

**Warum Cross-Validation statt des offiziellen Dev-Sets?**
- 63 Dev-Memes sind zu wenig, ein Meme bewegt den F1 um ca. 1 Punkt.
- 5-fold CV auf 750 Memes heißt: Jedes Meme wird genau einmal von einem Modell vorhergesagt, das es nie gesehen hat ("out-of-fold").

**Iterative Stratifikation.** Bei Multi-Label kann man nicht einfach nach *einer* Klasse stratifizieren. Der Algorithmus von Sechidis (2011) verteilt die seltenste Technik zuerst gleichmäßig auf die Folds, dann die nächste usw.

**Gepaarter Vergleich.**
- Alle Konfigurationen sehen exakt dieselben Folds. Wir vergleichen also Fold für Fold (Differenz ± Standardabweichung) und nicht zwei Mittelwerte.
- Das hat mehr statistische Power, weil "schwere" Folds für beide gleich schwer sind.

**Tie-Rule.** Eine Variante gewinnt nur, wenn ihr Vorsprung größer ist als seine Schwankung über die Folds. Sonst nehmen wir die einfachere. *Merksatz:* "Wir tunen nicht auf Rauschen."

**Test-Set nur einmal.** Alle Entscheidungen fallen auf den CV-Daten. Laut Rubrik bekommt man für eine "leaky in-distribution evaluation" im Judgment-Teil 0 Punkte.

**Bootstrap-Konfidenzintervall.** Man zieht 1000-mal mit Zurücklegen 200 Test-Memes, berechnet jedes Mal den F1 und nimmt das 2,5-%- und das 97,5-%-Perzentil. Das zeigt, wie stark der Testwert allein durch die Auswahl der 200 Memes schwankt.

## 3. Modell (§3)

**Tokenizer & `offset_mapping`.** Transformer sehen Subwort-Tokens ("BERN", "IE"), die Gold-Spans sind aber Zeichenpositionen.
- `offset_mapping` sagt für jedes Token, welche Zeichen es abdeckt (Start, Ende).
- Damit setzen wir Token-Labels (Training) und bauen aus Token-Vorhersagen wieder Zeichen-Spans (Auswertung).

**IO statt BIO.**
- BIO (Begin/Inside/Outside) aus der NER gibt jedem Token *ein* Label.
- Hier überlappen sich Techniken aber ständig (1379 Paare). Deshalb trifft jedes Token **20 Ja/Nein-Entscheidungen** (IO pro Technik).
- Laut unserer Prüfung kostet IO praktisch nichts: Das Round-Trip-F1 liegt bei 1,00.

**Multi-Task.** Ein Encoder, zwei Köpfe:
- Label-Kopf (Part A) und Span-Kopf (Part B), Loss = Label-Loss + Span-Loss.
- Bei wenig Daten hilft das, weil der Span-Loss dem Encoder sagt, *welche Wörter* wichtig sind.

**Label-wise Attention (unser selbst gebautes Segment).**
- Statt eines einzigen [CLS]-Vektors hat jede Technik einen eigenen gelernten Query-Vektor. Mit Softmax-Attention über die Tokens baut sie sich ihre eigene Zusammenfassung.
- So kann "Name calling" auf das Schimpfwort schauen und "Whataboutism" auf den "what about"-Satz.
- Vorbild: CAML (Mullenbach et al., 2018).

**Adaptionsstrategien (Woche 9).**
- *Frozen:* Nur die Köpfe werden trainiert.
- *LoRA:* Neben die Attention-Gewichte kommen kleine Matrizen A·B mit niedrigem Rang (r = 16). Trainiert wird nur ca. 1 % der Parameter.
- *Full Fine-Tuning:* Alles wird trainiert.

**Class Imbalance.**
- *pos_weight:* Positive Beispiele seltener Klassen zählen mehr. Wir nehmen die Wurzel, damit eine Klasse mit 4 Beispielen nicht 180-fach gewichtet wird.
- *Focal Loss:* Leichte, schon richtige Beispiele werden heruntergewichtet.

**Schwellen.** Das Modell gibt Wahrscheinlichkeiten aus, das Tool braucht Ja/Nein. Die Schwelle wird auf Out-of-Fold-Vorhersagen optimiert, nie auf dem Test-Set.

**Ensemble.** Die 5 Fold-Modelle werden gemittelt. Das reduziert Varianz, und ihre Schwellen sind auf genau diesem Modelltyp kalibriert.

## 4. Experimente E1–E9 (§4)

Jedes Experiment hat vorher eine **Hypothese** und eine **Entscheidungsregel**. Das zeigt den Markern, dass wir nicht herumprobiert, sondern gezielt getestet haben.
- **E9:** Zusatzdaten aus dem PTC (Propaganda in Nachrichtenartikeln).
  - Das ist erlaubt, weil wir selbst darauf trainieren und es deklarieren. Verboten wäre ein fremder Checkpoint, der schon auf Propaganda trainiert wurde.
  - Techniken, die das PTC nicht sauber annotiert, werden im Loss **maskiert**, also weder als positiv noch als negativ gewertet.

## 5. Explainability (§5, Woche 11–12)

**Integrated Gradients.**
- Man geht in kleinen Schritten von einer "neutralen" Eingabe (lauter Padding-Embeddings) zur echten Eingabe und summiert die Gradienten.
- Die Beiträge aller Tokens ergeben zusammen genau die Änderung des Logits ("Completeness").
- Unsere quantitative Frage: Liegt die Attribution im Gold-Span? Als Vergleich dient eine gleichmäßige Verteilung.

## 6. Part C

- **C1:** Der **kontrafaktische Tausch** (Trump ↔ Biden im *selben* Meme) ist der stärkste Test. Die Rhetorik bleibt gleich, nur die Seite ändert sich.
  - Gruppenvergleiche haben zu wenig Power (~160 vs. ~100 Memes, die sich auch inhaltlich unterscheiden).
  - Neutrale Templates prüfen, ob schon ein *Name allein* eine Technik auslöst.
- **C2:** Fairness des Vergleichs:
  - gleiche Memes, gleiche Metrik, gleiche Definitionen;
  - Antworten im falschen Format werden gezählt;
  - Wiederholungen wegen Temperatur 1;
  - Memorization-Test gegen Kontamination.
  - Wichtig: Ein Frontier-LLM darf im Einsatz gar nicht laufen, weil die Daten das Gerät verlassen würden.
- **C3:** Label-Kopf vs. Span-Kopf, 4 Fälle. Entscheidend ist, *wer bei Uneinigkeit recht hat*. Daraus leiten wir die Anzeige-Regel ab.
- **C4:** Es gibt keine australischen Labels. Deshalb:
  - **Transformationen gelabelter Testdaten** (Labels bleiben gültig): Namen auf AU umstellen, Kleinschreibung, Slang, Emoji, OCR-Fehler.
  - Ein kleines **Probe-Set** (von uns geschrieben, deklariert) für Fehlalarme bei harmlosen Teen-Memes.
  - Fun Fact: In den USA ist "liberal" links, in Australien sind die "Liberals" die konservative Partei. Genau so ein Fehler passiert in dieser Lücke.

## 7. Was du in einer mündlichen Nachfrage sagen können solltest

1. Warum Sigmoid und nicht Softmax?
2. Warum CV statt des offiziellen Dev-Sets, und warum ist das Test-Set trotzdem nicht "Australien"?
3. Was macht der Label-wise-Attention-Kopf anders als [CLS]?
4. Warum IO-Tagging, und was verliert man dabei?
5. Warum ist der LLM-Vergleich fair (oder wo ist er es nicht)?
6. Was zeigt der kontrafaktische Tausch, und was kann er *nicht* zeigen?
7. Würdest du das Tool einsetzen? Unter welchen Bedingungen?
