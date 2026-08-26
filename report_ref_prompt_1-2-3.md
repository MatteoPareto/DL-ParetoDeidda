# Report — Parti 1, 2 e 3 di `refactoring_prompt.md`

Eseguito il 2026-08-26 sulla VM DISI, branch `dev_vers_prova`.
Tutti gli indici di cella si riferiscono al notebook finale a **72 celle**.
Le Parti 4, 5 e 6 **non** sono state iniziate, come da istruzione.

Commit prodotti, in ordine:

| commit | parte |
|---|---|
| `4861a8d` | Parte 1 — correzioni testuali |
| `68353d0` | Parte 2 — §7 + rinomina + riesecuzione |
| `31a5379` | Parte 3 — assorbimento di METHOD.md |
| `58fa5f5` | Parte 2, seguito — tre riferimenti incrociati rimasti indietro |

Il quarto commit è separato perché l'ho trovato durante l'audit finale, dopo che la
Parte 3 era già chiusa. Non tocca output.

---

## 1. Sezioni toccate e cosa è cambiato

### Parte 1 — tre affermazioni che i dati smentivano

**§8.6 (cella 31).** La frase *"Prompt quality is a contributing floor"* era un'ipotesi mai
verificata. L'ho verificata: è falsa. I numeri non esistevano da nessuna parte nel repo, li
ho calcolati (n = 13 query uniche, Spearman con ranghi medi, permutazione a 50.000
rimescolamenti, bilaterale, seed fisso):

| relazione | rho | p |
|---|---|---|
| AUC minima della query vs R@10 (L3) | **+0,246** | 0,413 |
| AUC media della query vs R@10 (L3) | +0,149 | 0,623 |
| ceiling GT vs R@10 (L3) | **−0,747** | **0,0049** |
| ceiling GT vs R@10 (L1) | −0,074 | 0,812 |
| AUC minima vs ceiling GT | +0,030 | 0,923 |

Tutto ciò che il piano asseriva è confermato. Il testo nuovo riporta il risultato come
negativo esplicito: la difficoltà per query è governata dalla concentrazione del ground
truth, non dalla qualità del prompt, e `-Young` (0,205, ceiling 0,418) contro
`+Chubby, -Young` (0,560, ceiling 0,067) lo mostra a parità di attributi.

**§8.4 (cella 31).** Riscritta attorno al riferimento corretto. La tabella ora ha la colonna
*useful diversity*, la riga di A e la riga dell'oracolo, lette da `collapse_diag.pt`:

| metodo | frac_distinct | useful diversity |
|---|---|---|
| L2 | 0,757 | 0,018 |
| L1 | 0,728 | 0,017 |
| L3 | 0,257 | 0,102 |
| A | 0,230 | 0,222 |
| oracolo | 0,281 | 0,903 |

Ho aggiunto l'aritmetica per query su `+Eyeglasses`, la spiegazione del meccanismo
(selettività mediana **13,3%** — calcolata da `list_attr_celeba.txt` sullo split di test,
non ricopiata dal piano — combinata con Hamming ≤ 2 sui 38 attributi non vincolati), e la
conferma indipendente data da A. Una nota dichiara che la tabella live della §10.1 media su
14 voci invece che 13 e quindi differisce sul terzo decimale: era una piccola incoerenza
già presente, ora è dichiarata invece che nascosta.

**§10.5 (cella 48).** Riletta col bersaglio giusto: l'ancora non scambia recall per varietà,
sposta la concentrazione. Ho aggiunto la regola di selezione priva di leakage e il motivo
misurato per cui non si applica (oracolo di validation 0,4155 contro 0,2815 sul test).
`lambda = 0,1` **non** è stato adottato.

### Parte 2 — §7 descriveva un modello che non è quello misurato

**§7 (cella 23), riscritta.** Il titolo era *"Gated cross-attention fusion"* e il corpo dava
la derivazione formale di cross-attention e sign embeddings, entrambi disattivati in
`FUSION_CFG`. Ora §7.1 descrive il forward pass deployato, che ho verificato riga per riga
contro `GatedCrossAttentionFusion.forward` invece di fidarmi dello pseudocodice del piano:

```
c     = cond_proj(cond_emb * s)      s = +1 aggiungi, -1 rimuovi
c     = self_attn(c)
delta = out_proj(mean_pool(c))
g     = sigmoid(MLP([v_ref ; delta]))
q     = normalize(v_ref + g * delta)
```

Il callout con la conseguenza strutturale — `delta` non dipende da `v_ref`, il gate è
l'**unico** componente dipendente dalla sorgente, quindi il modello è una direzione appresa
per query la cui intensità è modulata dall'immagine di riferimento — è al suo posto subito
dopo l'architettura, prima di ogni altra cosa.

**§7.4.** La tabella iperparametri era anch'essa ferma al modello progettato: diceva
*~4M params* e *lr 1e-4*. Il modello deployato è **3,02M a 3e-4**. Corretta. Il piano non
lo chiedeva; l'ho fatto perché è esattamente lo stesso difetto che la Parte 2 esiste per
riparare.

**§10 (cella 40).** Ha ricevuto la storia del design: i quattro meccanismi con la tabella
F1–F5 completa, e il paragrafo che dice apertamente che tre dei quattro sono stati
falsificati.

**Rinomina.** `"L3 gated cross-attention"` → `"L3 gated residual fusion"`, 9 occorrenze nel
sorgente. `FUSION_CFG` non è stato toccato, quindi nessuna cache di studio è stata
invalidata.

**Bug preesistente corretto.** La fine di §7 conteneva `\n` letterali (non a capo) da un
edit precedente: quattro righe che nel notebook renderizzato apparivano come testo grezzo.

**Conclusioni, punto 4 (cella 71).** Leggeva ancora la diversità contro il ~75% di L1/L2.
Aggiornato all'oracolo. Non era nella lettera del piano, ma è la stessa affermazione che la
Parte 1.2 corregge in §8.4: lasciarla avrebbe messo il notebook in contraddizione con sé
stesso a due sezioni di distanza.

### Parte 3 — METHOD.md assorbito

**Conteggio celle: 69 prima, 69 dopo.** Nessuna cella aggiunta. Tutto è entrato in celle
markdown esistenti (23 e 68), con blocchi in grassetto invece di nuove sottosezioni
numerate, così la numerazione dell'indice resta stabile.

Dentro **§7** (cella 23): motivazione dello zero-init (§2.4, incluso il dettaglio del
gradiente nullo al primo passo), motivazione del gate e della rinormalizzazione (§2.5, con
la differenza rispetto a TIRG), la tabella **alternative considerate e scartate** (§4), le
tre decisioni che rendono onesta la supervisione minata e l'argomento no-leakage (§3.1), il
perché di InfoNCE / τ = 0,07 / due tipi di negativi (§3.2), e la colonna *why* nella tabella
iperparametri (§3.3).

Dentro **§12** (cella 71): la bibliografia ora dice **cosa** ogni lavoro ha dato al progetto,
in tre tabelle (spazio/dataset, componenti classici, lavori adiacenti). Ho aggiunto tre
citazioni che erano usate nel design ma mancavano dalla bibliografia: Hochreiter &
Schmidhuber (LSTM), Zhang et al. (ControlNet, zero-convolutions), Robinson et al. (hard
negatives). Dichiarazione di provenienza portata alla forma completa.

Il notebook non rimanda più a `METHOD.md`, `RUNNING.md` né `HANDOFF.md` — verificato con
grep su tutte e 69 le celle. I tre file restano nel repo come documenti di processo.

---

## 2. Riferimenti incrociati aggiornati

L'audit su `Section 7|§7` ha dato 16 occorrenze in 13 celle. Tre erano diventate false, e
tutte per lo stesso motivo: l'architettura *progettata* si è spostata da §7 a §10.

| cella | sezione | prima | dopo |
|---|---|---|---|
| 31 | §8.3 | "Ablations against the architecture as designed in Section 7" | → Section 10 |
| 31 | §8.3 | "the single design argument from Section 7 the data supports" | → "the original design" |
| 43 | §10.2 (commento) | "DESIGNED_CFG is Section 7's proposal" | → "the original design (Section 10)" |

Le altre 13 sono rimaste corrette e le ho verificate una per una: celle 0, 5, 7, 20 citano
§7 per il mining e il training; 31 cita §7.2 per il caso ad attributi correlati; 55, 64
citano §7.2 per la costruzione degli esempi; 56, 57 citano §7 per la proprietà CLAY del
database congelato. §7.2 non ha cambiato numero né contenuto, quindi tutti i rimandi a §7.2
sono sopravvissuti alla riscrittura.

Aggiornati anche, fuori dal perimetro §7:
- cella 40 (§10) — il punto 2 non rimanda più a "Section 7.1 motivates four mechanisms",
  perché ora quei quattro meccanismi sono lì sotto;
- cella 71 — punto 4 delle conclusioni (vedi sopra);
- celle 1, 3, 4 — rimossi i rimandi a `RUNNING.md`. Nel farlo la cella 1 mi era rimasta con
  una frase troncata ("Step-by-step" senza seguito); l'ho riscritta.

---

## 3. Riesecuzione e coerenza output/codice

**Sì, rieseguito.** Giro completo con `jupyter nbconvert --execute`, tutte e 69 le celle,
dopo la Parte 2. Durata ~20 minuti con le cache calde.

- **0 errori**, 0 celle non eseguite.
- Il nome vecchio non compare più in **nessun** output. Il nome nuovo compare negli output
  delle celle **30, 32, 33, 39, 41, 53, 59, 65**.
- **Ogni singolo valore numerico in ogni output è identico** al giro precedente. L'ho
  verificato meccanicamente estraendo tutti i float da tutti gli output di tutte le celle e
  confrontando cella per cella: zero differenze. La riesecuzione ha riprodotto i risultati
  esattamente, il che conferma anche che nessuna cache è stata invalidata.
- Unica altra differenza: la cella 69 è passata da 45 a 3 output, perché le 42 barre di
  avanzamento erano il residuo di un giro a cache fredda. I numeri della §11.9 sono
  identici. Il file è per questo 475 KB più piccolo.

**Le Parti 1 e 3 non hanno richiesto riesecuzione**, e l'ho verificato invece di assumerlo:
dopo entrambe ho confrontato gli output contro HEAD e sono risultati byte-identici. Le due
celle di codice toccate dalla Parte 3 (3 e 4) contengono solo un commento e un messaggio di
`assert` che scatta unicamente se manca l'archivio del dataset — su questa VM non è mai
stato raggiunto.

---

## 4. Cose su cui non sono d'accordo, o che non ho potuto fare

### Divergenze dal piano, adottate

**Il piano diceva che le Parti 1–3 non richiedono di eseguire nulla. Per la Parte 1 non è
vero**, ed è una seconda imprecisione oltre a quella che avevi già segnalato tu. La §1.1
dice "verifica tu i numeri" e quei numeri non esistevano nel repo: li ho dovuti calcolare.
Non è una riesecuzione del notebook e non tocca cache, ma non è zero calcolo. (Il venv non
ha `scipy`, quindi Spearman e il test di permutazione sono implementati a mano.)

**Due numeri del piano sono sbagliati di uno.** Il piano dice che su `+Eyeglasses` L1
restituisce 2.876 immagini distinte e che almeno 2.178 sono sbagliate. `collapse_diag.pt`
dice 2.875, e 2.875 − 698 = **2.177**. Ho scritto i valori del file, non quelli del piano.

**p = 0,005 invece di 0,004.** Il piano dà p = 0,004 per la correlazione ceiling↔recall; la
mia misura dà 0,0049. È rumore di permutazione, ma ho scritto 0,005 perché è quello che ho
misurato.

**Non ho scritto che λ = 0,1 dista 0,006 dall'oracolo.** Il piano lo presenta come uno scarto
preciso. Non regge: lo sweep dell'ancora usa **due soli seed**, e la diversità è essa stessa
dipendente dal seed — riaddestrare la configurazione deployata (`lambda = 0`) dà 0,234
contro lo 0,257 dell'istanza deployata misurata in §8.4. Uno spread di 0,023 è quasi quattro
volte lo scarto rivendicato. Ho scritto la **direzione** del movimento, che è il punto
sostanziale e regge, e ho messo il caveat sulla precisione nero su bianco in §10.5. Se
preferisci la formulazione originale è una riga da cambiare, ma io non la firmo.

### Imprecisione nel piano, fuori dal mio perimetro

La **Parte 4.3** dice che §10.7 e §10.8 vivono in `section_10_7_cells.py` e
`section_10_8_cells.py`, "scritti ma mai inseriti né eseguiti". **Quei file non esistono nel
repo**, e **§10.7 è già nel notebook** (celle 52–53) con output. Quando arriverai alla Parte
4, quella voce va rifatta: resta aperta solo §10.8.

### Cose che non ho fatto

- **Parti 4, 5 e 6**: non iniziate, come da istruzione.
- **`METHOD.md` è ora in parte duplicato dentro il notebook** e il suo titolo è ancora
  "Level 3 — Gated Cross-Attention Fusion", cioè il nome che la Parte 2 ha ritirato. Il
  piano dice che resta come documento di lavoro e l'ho lasciato intatto. Ma se qualcuno lo
  apre dopo la rinomina trova un nome che non esiste più: varrebbe una riga di STATUS in
  testa. Non l'ho aggiunta perché non me l'hai chiesta.
- **La verifica su Colab (Parte 6.4) resta il rischio aperto principale**, invariato rispetto
  a prima: il notebook non è mai stato eseguito lì. La riesecuzione di oggi conferma però
  che a cache calde tutto ricarica senza riaddestrare nulla, che è la precondizione perché
  il giro Colab sia praticabile in ~10 minuti.
- **`refactoring_prompt.md` è rimasto untracked** e non è in nessuno dei quattro commit. Mi
  hai detto che può anche finire su git; se lo vuoi dentro, è un `git add` separato.

---

# Addendum — due correzioni alla Parte 2 (commit `8d6eefc`)

Arrivate dopo che la Parte 2 era già chiusa, quindi applicate come commit sopra.

## 1. La frase perno del §7 ora cita la misura, non il forward pass

**I tre numeri sono verificati** nell'output della cella 53, come chiesto: `retained`
0,517 (permutato) → 52%, 0,622 (centroide) → 62%, e R@10 del centroide 0,210 contro 0,110
di L1, cioè 1,9×. Tutti corretti.

**Un pezzo della frase proposta però non torna.** Diceva *"roughly two thirds of L3's
advantage over L1"*. Il 62% è la quota del **recall** di L3 che sopravvive senza
informazione sulla sorgente (0,210 / 0,338), non la quota del suo **vantaggio su L1**:

```
vantaggio di L3 su L1        0,338 - 0,110 = 0,228
vantaggio del centroide      0,210 - 0,110 = 0,100
quota del vantaggio          0,100 / 0,228 = 44%
```

Ho tenuto la tua struttura e la tua formulazione, sostituendo la sola clausola finale con
il dato esatto: *"a large part of L3's advantage over L1 comes from having learned a better
per-query direction… that component accounts for 0.100 of the 0.228 that separates L3 from
the baseline."* Il 62% resta nel testo, attaccato alla quantità giusta.

Ho aggiunto anche che il calo è significativo su **10 query su 13**, che è nell'output della
cella 53 e rafforza la prima delle due letture.

## 2. Numerazione

**§10.** 10.1, 10.2 e 10.3 esistevano solo come commenti nel codice. Ora hanno header
markdown: 10.1 accodato alla cella 40 esistente, 10.2 e 10.3 in una cella header ciascuna.
L'indice va da 10.1 a 10.7 senza salti.

**§11.** Numerata la 11.1 ("What the two diagnostics say"); aggiunto l'header della 11.2
— che tra l'altro la bibliografia cita già per i linear probe — accodandolo alla stessa
cella; aggiunto l'header della 11.6 in una cella nuova.

**11.8 / 11.9: ho rinumerato invece di riordinare.** Spostare fisicamente la 11.8 dopo la
11.9 avrebbe lasciato la sequenza 11.7, 11.9, 11.8. E la 11.8 è *"What Section 11 changes"*,
cioè il riepilogo della sezione: va in fondo. Quindi ho scambiato i numeri — lo studio sulla
supervisione ausiliaria diventa 11.8, il riepilogo diventa 11.9 — e l'ordine risulta
corretto sia numericamente sia logicamente, senza muovere nessuna cella. Aggiornati il
rimando interno dentro lo studio, il tag nel commento del codice, e i **quattro** riferimenti
in `HANDOFF.md`, che altrimenti sarebbero rimasti a puntare al contrario.

## Conteggio celle e riesecuzione

**69 → 72.** Le tre celle aggiunte sono soli header (10.2, 10.3, 11.6); 10.1 e 11.2 sono
entrati in celle già esistenti. Non è stato possibile stare a 72 − 3 = 69: perché una
sottosezione compaia nell'indice serve un header markdown, e quelle tre sono precedute da
celle di codice, non da markdown in cui accodarlo. Se il vincolo "niente celle nuove" vale
anche qui invece che solo per la Parte 3, dimmelo e le tolgo — ma allora 10.2, 10.3 e 11.6
restano fuori dall'indice.

**Nessuna riesecuzione.** Gli output di tutte le celle preesistenti sono byte-identici
(confrontati per `id`, non per posizione, perché le posizioni sono cambiate), gli
`execution_count` restano monotoni, e l'unica modifica a codice è il retag di un commento.

## Rimane un header senza numero

`### The architecture as designed, and the failure modes it answered`, nella cella 40. È il
blocco che la Parte 2.3 ha spostato lì da §7. L'ho lasciato senza numero di proposito: sta
prima della 10.1 ed è preambolo della sezione, e numerarlo 10.1 avrebbe fatto slittare di
uno tutte le sottosezioni di §10 con i relativi rimandi incrociati. Se lo vuoi numerato è
fattibile, ma è una rinumerazione a catena, non dieci minuti.

---

# Addendum 2 — riverifica completa (commit successivo a `8d6eefc`)

Riletto tutto. Il grosso regge; ho trovato **cinque** cose, quattro mie e una preesistente.

## Verifiche superate

- **Nessun riferimento pendente.** Ogni `Section N.M` / `§N.M` nel notebook risolve a un
  header che esiste: 1–12, 7.1–7.5, 8.1–8.6, 10.1–10.7, 11.1–11.9. Zero orfani.
- **Tabella §8.4 contro `collapse_diag.pt`**: tutte e dieci le cifre combaciano.
- **Aritmetica su `+Eyeglasses`**: 698 bersagli, 4.000 caselle, 2.875 distinte, 2.177
  certamente sbagliate. Tutti confermati dal file.
- **Callout §7 contro l'output della cella 53**: 0,517 / 0,622 / 0,210 e le 10 query su 13
  significative. Tutti confermati.
- **Header nuovi contro il codice che precedono**: 10.1, 10.2, 10.3, 11.2, 11.6 descrivono
  ciascuno ciò che la cella successiva effettivamente fa (verificato anche il "cinque seed"
  della 10.3 contro `SEEDS = [SEED + i for i in range(5)]`).
- **Coerenza dei numeri di diversità** fra §8.4, §10.5, §10.7, §11 e le conclusioni:
  ~26% / 0,257 / 0,281 / 0,234 sono usati ciascuno nel posto giusto.

## Corretto adesso — quattro sviste mie

1. **"nearly three times the perfect ranker"** in §8.4, e **"three times above"** nelle
   conclusioni. Il rapporto vero è **2,59× (L1)** e **2,69× (L2)**. Avevo introdotto io
   un'esagerazione retorica nella sezione il cui scopo è togliere le affermazioni non
   verificate. Ora dice `2.6x and 2.7x` e "more than two and a half times".

2. **§7.5 elencava ancora i sign embeddings fra i componenti del modulo.** Questo è
   l'errore più serio dei quattro: è esattamente il difetto che la Parte 2 esiste per
   eliminare, sopravvissuto in fondo alla stessa sezione che ho riscritto. Ora §7.5
   descrive i componenti del modello **deployato**, e i due meccanismi ablati (sign
   embeddings, cross-attention) sono attribuiti insieme alla storia del design in §10.
   Nessuna citazione è andata persa.

3. **L'abstract diceva che il modulo "weighs the conditions".** Il modello deployato fa
   mean pooling, quindi pesi uniformi: è la pesatura dipendente dalla sorgente che la
   cross-attention forniva ed è stata ablata. Ora dice "learns a direction from the
   conditions and controls how far to move along it with a learned gate".

4. **Frase goffa in §8.3** creata dal mio commit `58fa5f5`: *"the single design argument
   from the original design"*. Ora *"the single argument from the original design that the
   data supports"*.

## Trovato ma NON corretto — decide chi scrive il report

**§8.3 si contraddice sulla self-attention.** Dice:

> Self-attention is neutral (-0.004, inside noise) **while halving the parameter count**;
> it is retained because **it costs nothing** and Section 10.6 shows no benefit to dropping it.

Le due affermazioni non stanno insieme, e la tabella sopra dà ragione alla prima:

```
designed 4,07M  ->  senza self-attention 1,97M   = 2,10M risparmiati
deployed 3,02M  ->  senza self-attention ~0,92M
la self-attention è circa il 70% del modulo deployato
```

Toglierla costa −0,004 di R@10, dentro il rumore, e fa risparmiare **il 70% dei
parametri**. "It costs nothing" è falso sull'asse dei parametri. È un difetto
**preesistente**, non l'ho introdotto io, e non l'ho toccato perché la riparazione richiede
di decidere cosa volete sostenere: che la self-attention si tiene perché è gratis (falso),
o che si tiene nonostante costi, perché la 10.6 non mostra vantaggi a toglierla e l'ablation
è dentro il rumore (vero, ma è una tesi diversa). Dimmi quale e la scrivo.

## Indici di cella nel corpo del report

Erano rimasti alla numerazione a 69 celle, prima delle tre inserzioni. Riallineati tutti
alla numerazione finale a 72 celle: §10.5 46→48, conclusioni 68→71, commento DESIGNED_CFG
42→43, output della §10.7 51→53, celle rieseguite 51/57/62→53/59/65, cella dell'aux 66→69.
