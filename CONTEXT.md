# Contesto Completo del Progetto — Deep Learning 2026
## Image Retrieval su CelebA con CLIP e Composizionalità

---

## 1. OBIETTIVO DEL PROGETTO

Il task è **Compositional Image Retrieval su CelebA**: dato un'immagine di riferimento e una query testuale che specifica attributi da aggiungere o rimuovere (es. `+glasses, -smile`), il sistema deve recuperare le immagini più simili che soddisfano i vincoli richiesti.

Il progetto si divide in due fasi:
1. **Baseline zero-shot**: aritmetica nello spazio latente CLIP (`v_target ≈ v_ref + t_glasses - t_red_hair`)
2. **Metodo avanzato**: un meccanismo di fusione più sofisticato che supera i limiti del baseline

---

## 2. DATASET: CelebA

- Dataset di volti di celebrità con **40 attributi binari** annotati (es. Smiling, Eyeglasses, Male, Blond Hair, ecc.)
- Setup su Google Colab: mount di Google Drive + unzip del dataset
- Gli embedding visivi vanno estratti **offline una sola volta** con CLIP e tenuti frozen

---

## 3. MODELLO DA USARE

- **CLIP ViT-B/32** da HuggingFace: `openai/clip-vit-base-patch32`
- Gli embedding hanno dimensione **512**
- Embedding immagine: `u_I ∈ ℝ^512`, embedding testo: `u_T ∈ ℝ^512`
- La similarità è misurata con **cosine similarity**
- Si può testare altri modelli, ma ViT-B/32 è quello obbligatorio da riportare

---

## 4. PIPELINE RACCOMANDATA

1. **Data exploration**: familiarizzare con le annotazioni di attributi di CelebA
2. **Feature extraction offline**: estrarre e salvare gli embedding CLIP per tutte le immagini del corpus
3. **Baseline zero-shot**: implementare l'aritmetica latente naïve (senza SVD né fusione)
4. **Sviluppo del metodo**: implementare un meccanismo di fusione avanzato e confrontarlo con la baseline

---

## 5. VALUTAZIONE OBBLIGATORIA

### Query mandate (da `celeba_evaluation.json` su Google Drive):

**Query semplici (un attributo):**
- `+ Smiling`
- `+ Eyeglasses`
- `- Heavy Makeup`
- `+ Male`
- `- Young`
- `+ Blond Hair`
- `+ Mustache`

**Query composte (più attributi):**
- `+ Eyeglasses & - Smiling`
- `- Male & - Mustache`
- `+ Chubby & - Young`
- `- Smiling & + Eyeglasses & + Wearing Hat`
- `+ Wearing Lipsticks & - Heavy Makeup & + Smiling`

### Ground Truth
Il ground truth è basato su una **distanza di Hamming rilassata**: un'immagine è valida se:
1. Soddisfa i vincoli +/- della query
2. Ha Hamming distance ≤ 2 dagli attributi non specificati dell'immagine di riferimento

Solo le immagini sorgente con **≥ 5 ground truth validi** sono incluse nel benchmark.

### Metriche
Per ogni query, calcolare a **K = 1, 5, 10** (media su tutte le immagini sorgente valide):
- **Recall@K** (metrica primaria): `1` se almeno un ground truth è nei top-K risultati, `0` altrimenti
- **Precision@K** (secondaria): `|R_K ∩ G| / K`

---

## 6. SFIDE TECNICHE PRINCIPALI

Il progetto si concentra su due problemi chiave:

### 6.1 Composizionalità multi-attributo
CLIP ha **limiti di composizionalità**:
- Tende a comportarsi come un modello *bag-of-words* in modalità cross-modal (non rispetta l'ordine delle parole, ha difficoltà con binding attributo-oggetto, relazioni spaziali, negazione)
- Funziona bene per attributi singoli, ma degrada su combinazioni complesse

### 6.2 Superamento della bottleneck SVD naïve
La semplice concatenazione di embedding testuali e visivi prima di SVD è limitante. Il progetto richiede meccanismi più avanzati, come:
- Cross-attention layers
- Gating mechanisms
- Projection heads non lineari
- Contrastive alignment strategies

---

## 7. CONSEGNA

- **Un singolo Jupyter Notebook** su Google Colab, self-contained
- Il notebook deve contenere: codice + report in Markdown

### Il report deve includere:
- **Descrizione metodologica** (con matematica formale: architettura, forward pass, loss se applicabile)
- **Setup sperimentale** (motivazione delle scelte di design, ottimizzatore, iperparametri)
- **Risultati e discussione** (tabelle comparative Recall@K, grafici learning curve se applicabile, esempi qualitativi di successi e fallimenti)

### Criteri di valutazione:
- Originalità e creatività del meccanismo di fusione
- Rigore metodologico
- Chiarezza del report
- Performance empiriche (R@1, R@5 vs baseline)
- Qualità del codice (leggibilità, modularità, efficienza)

---

## 8. CONCETTI TEORICI CHIAVE

### 8.1 CLIP (Radford et al., ICML 2021)
- Pre-training contrastivo su 400M coppie (immagine, testo) da internet
- Training objective: massimizzare la cosine similarity tra embedding di coppie corrette, minimizzarla per quelle incorrette
- **Zero-shot transfer**: genera classificatori da nomi di classe testuali senza esempi di training
- **Prompt engineering**: usare `"a photo of a {label}"` invece del solo nome migliora le performance
- Gli embedding sono normalizzati su una **ipersfera unitaria** (geometria sferica, non euclidea)

### 8.2 Composizionalità (Berasi et al., CVPR 2025 — GDE)
- La composizionalità è il principio per cui il significato di un'espressione complessa emerge dai suoi componenti primitivi
- CLIP mostra strutture composizionali: gli embedding sono **approssimativamente decomponibili** geodesicamente
- Formula lineare: `u_(attr,obj) ≈ u_attr + u_obj` (Trager et al., ICCV 2023)
- Formula geodesica (più corretta per la geometria sferica): `u_(attr,obj) = Exp_μ(v_attr + v_obj)`
- Le direzioni primitive si trovano come **medie tangenti** nel tangent space centrato sulla media intrinseca μ
- Il framework GDE (Geodesically Decomposable Embeddings) funziona su dati visivi rumorosi e sparsi
- Risultati: GDE supera la decomposizione lineare in classificazione composizionale e group robustness (stato dell'arte su CelebA e Waterbirds)

### 8.3 CLAY (Lim et al., CVPR 2026)
- **Conditional Visual Similarity Modulation**: modifica lo spazio di similarità invece degli embedding stessi
- **Training-free**: nessun fine-tuning richiesto
- Costruisce un **textual subspace** tramite SVD sulle feature testuali della condizione (con logarithm map per rispettare la geometria iperbolica)
- Proietta gli embedding visivi su questo sottospazio condizionale → similarità coseno nel sottospazio
- Supporta **multi-conditional retrieval** (più condizioni simultanee) senza ricalcolare gli embedding del database
- Molto più efficiente di GeneCIS e altri metodi training-based
- **Ispirazione diretta per il progetto**: il progetto chiede di andare oltre la SVD naïve di CLAY

### 8.4 Proprietà geometriche dello spazio latente CLIP
- **Narrow Cone Effect**: gli embedding si concentrano in un cono stretto (avg cosine ~0.5, cioè ~60°). In ℝ^512, questo cono occupa ~10^{-153}% della superficie dell'ipersfera
- **Modality Gap**: il cono delle immagini e il cono dei testi sono separati → attenzione quando si combinano embedding cross-modali
- **Ortogonalità e Superposition**: concetti di attributi diversi sono approssimativamente ortogonali. In ℝ^N si possono fitting ~exp(εN) vettori quasi-ortogonali (lemma Johnson-Lindenstrauss)
- **Log e Exp maps**: per operare correttamente sulla sfera bisogna usare le mappe logaritmica ed esponenziale invece dell'aritmetica euclidea

---

## 9. POLICY DEL PROGETTO

- Gruppi di 2-3 studenti (singoli ammessi ma sconsigliati)
- **Vietato** usare repository GitHub completi di terze parti come base
- Il codice va costruito da zero (o a partire dallo skeleton fornito)
- Snippet specifici da open-source sono ammessi solo se commentati e citati
- Non condividere codice tra gruppi → plagiarism detection automatico
- La policy vale anche per chi "presta" il codice

---

## 10. RIFERIMENTI PRINCIPALI

| Paper | Rilevanza |
|---|---|
| Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021 | CLIP — modello base del progetto |
| Berasi et al., "Not Only Text: Exploring Compositionality of Visual Representations in VLMs", CVPR 2025 | GDE — composizionalità visiva, usata nel progetto |
| Lim et al., "CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space", CVPR 2026 | CLAY — metodo di riferimento da superare |
| Trager et al., "Linear Spaces of Meanings: Compositional Structures in VLMs", ICCV 2023 | Fondamenta teoriche della decomposizione lineare |
| Liu et al., "Deep Learning Face Attributes in the Wild", ICCV 2015 | Dataset CelebA |
| Yuksekgonul et al., "When and why VLMs behave like bag-of-words", ICLR 2023 | Limiti composizionalità CLIP |

---

## 11. NOTE PRATICHE

- Usare Google Colab con GPU
- Estrarre gli embedding **una sola volta** e salvarli (es. su Drive) per efficienza
- Il file `celeba_evaluation.json` su Drive contiene gli indici esatti delle immagini da usare per la valutazione → **non ignorarlo**
- Le query di valutazione sono fisse e pubbliche: si può usarle per guidare lo sviluppo
- La baseline zero-shot (aritmetica diretta) è il lower bound da battere