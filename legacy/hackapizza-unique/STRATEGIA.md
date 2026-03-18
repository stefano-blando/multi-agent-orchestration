# Hackapizza 2.0 — Strategia

## Dati di partenza
- 25 team, 287 ricette, 62 ingredienti unici
- Balance iniziale: 1000
- Ingredienti scadono a fine turno
- Turni da ~5-7 minuti, fasi: speaking → closed_bid → waiting → serving → stopped
- Golden Run domenica 10-12: solo agenti, nessun intervento umano

---

## 1. Selezione ricette (gold recipes)

**Status: ANALIZZATO — da implementare nel prompt**

Strategia "nicchia di lusso": 3 piatti con ingredienti poco contesi.

| Piatto | Prestige | Ingr | Prep | Target prezzo |
|--------|----------|------|------|---------------|
| Portale Cosmico... | 100 | 5 | 5.2s | 150-200 |
| Sinfonia Temporale... | 95 | 5 | 4.0s | 120-160 |
| Multiverso Calante | 85 | 5 | 4.0s | 70-100 |

Solo 13 ingredienti unici. 7 a bassa/media competizione.
Costo bid stimato: ~100-130 su 1000 di balance.

### Domande aperte
- [ ] Servono piatti cheap per esploratori?
- [ ] Il 4° piatto è un rischio o un vantaggio?
- [ ] I prezzi vanno calibrati dopo i primi turni di test

---

## 2. Strategia bid

**Status: DA FARE**

### Bid guidati per competizione
| Competizione ingrediente | Bid suggerito per unità |
|--------------------------|------------------------|
| Bassa (< 30 ricette) | 3-5 |
| Media (30-40 ricette) | 5-8 |
| Alta (40+ ricette) | 8-15 |

### Idee
- [ ] **Budget cap**: non spendere più del 40% del balance in bid
- [ ] **Bid adattivo**: se ai turni precedenti abbiamo vinto tutto a bid basso, abbassare ancora. Se abbiamo perso, alzare
- [ ] **Overbid selettivo**: sugli ingredienti chiave (Uova di Fenice) biddare alto per sicurezza, su quelli di nicchia biddare il minimo
- [ ] **Bid history**: dopo ogni turno leggere GET /bid_history per capire quanto biddano gli altri e adattarsi

### Domande aperte
- [ ] Qual è il prezzo di mercato degli ingredienti? Lo scopriamo solo dal bid_history
- [ ] Conviene biddare su ingredienti extra da rivendere al mercato?

---

## 3. Memoria tra turni

**Status: DA FARE**

L'agente ora non ricorda nulla tra un turno e l'altro. I giudici valutano "long term planning" e "memory".

### Cosa salvare tra turni
- [ ] Bid history: quanto abbiamo biddato e quanto hanno biddato gli altri
- [ ] Prezzi di mercato medi per ingrediente
- [ ] Quali piatti sono stati ordinati (domanda clienti)
- [ ] Quali piatti abbiamo servito con successo e ricavo
- [ ] Reputazione attuale
- [ ] Pattern dei competitor (chi bidda su cosa)

### Come implementare
- File JSON locale? (`memory.json` aggiornato a fine turno)
- Oppure campo in GameState che persiste?
- L'agente deve leggere la memoria a inizio turno nel prompt

### Domande aperte
- [ ] Quanti token occupa la memoria nel prompt? Rischio di sforare
- [ ] Meglio riassumere o dati grezzi?

---

## 4. Gestione serving

**Status: DA FARE — critico per la Golden Run**

Il serving è dove facciamo i soldi. Errori qui = perdita diretta.

### Problemi da risolvere
- [ ] **Match ordine → piatto**: il client orderText è testo libero, va interpretato
- [ ] **Intolleranze**: le rules dicono che servire cibo sbagliato = penalità. Come sappiamo le intolleranze? Arrivano nell'orderText? Nel client data?
- [ ] **Coda clienti**: se arrivano 5 clienti insieme, servire in ordine. Se non abbiamo ingredienti, meglio non servire che servire male?
- [ ] **prepare → wait → serve**: c'è un delay tra prepare_dish e preparation_complete. L'agente deve aspettare l'evento prima di servire
- [ ] **Chiusura tattica**: chiudere il ristorante se stiamo finendo ingredienti, per evitare clienti che non possiamo servire

### Flusso ideale
1. Arriva client_spawned con orderText
2. Agente matcha orderText con un piatto del menu
3. Controlla se abbiamo ingredienti
4. prepare_dish() → aspetta preparation_complete
5. serve_dish() al client_id
6. Se non possiamo servire → skippa il cliente (meglio non servire che servire male?)

### Domande aperte
- [ ] Possiamo servire più piatti in parallelo? (kitchen = [])
- [ ] Cosa succede se non serviamo un cliente? Penalità reputazione?
- [ ] Il tempo di serving phase è fisso o dipende dai clienti?

---

## 5. Mercato (market trading)

**Status: DA FARE**

Il mercato è un'opportunità sottovalutata. Le rules dicono che è pubblico e che "essere reattivi può portare a grandi guadagni".

### Strategie possibili
- [ ] **Vendita surplus**: a fine waiting, vendere ingredienti che non useremo (scadono comunque)
- [ ] **Acquisto opportunistico**: se qualcuno vende cheap un ingrediente che ci serve, comprare
- [ ] **Speculazione**: comprare ingredienti contesi e rivenderli a prezzo maggiorato
- [ ] **Sabotaggio gentile**: creare entry BUY a prezzo bassissimo per confondere
- [ ] **Monitoraggio**: controllare il mercato periodicamente durante serving per occasioni last-minute

### Domande aperte
- [ ] Il prezzo è totale o per unità? (è TOTALE, confermato dallo schema)
- [ ] Quanti team usano davvero il mercato? Se pochi, poco utile
- [ ] Vale la pena spendere step dell'agente sul mercato vs servire clienti?

---

## 6. Diplomazia (messaging)

**Status: DA FARE — bassa priorità ma punti bonus**

Le rules menzionano alleanze e tradimenti. I giudici valutano "relazioni strategiche".

### Idee
- [ ] **Alleanze su ingredienti**: "noi biddiamo su X, voi su Y, poi ci scambiamo al mercato"
- [ ] **Info trading**: condividere bid_history in cambio di info
- [ ] **Bluff**: dire che biddiamo su ingredienti che non ci servono
- [ ] **Auto-messaggio**: mandare messaggi automatici a inizio partita per sembrare collaborativi

### Domande aperte
- [ ] Vale la pena investire tempo su questo vs ottimizzare le altre cose?
- [ ] Come gestire se un team ci propone un accordo? L'agente deve capirlo dal testo

---

## 7. Analisi competitor

**Status: DA FARE**

GET /restaurants ci dà overview di tutti. Possiamo tracciare.

### Cosa monitorare
- [ ] Balance degli altri team (chi sta vincendo?)
- [ ] Menu degli altri (GET /restaurant/:id/menu per quelli aperti?)
- [ ] Reputazione (chi serve bene, chi male)
- [ ] Chi è aperto/chiuso
- [ ] Bid history (chi bidda quanto su cosa)

### Come usare
- [ ] Se un team domina un ingrediente, evitarlo
- [ ] Se tutti puntano su ricette di alto prestigio, noi andiamo su volume/budget
- [ ] Adattare prezzi: se i competitor prezzano basso, o ci adattiamo o puntiamo su qualità

---

## 8. Resilienza e edge case

**Status: DA FARE — critico per Golden Run**

La Golden Run è 2 ore senza intervento. L'agente deve reggere da solo.

### Problemi da gestire
- [x] SSE reconnect (fatto)
- [x] Agent in thread separato (fatto)
- [x] Logging su file (fatto)
- [ ] **Rate limiting**: il server ha rate limit (6/s GET, 20/s MCP). L'agente non deve spammare
- [ ] **Errori MCP**: gestire risposte con isError=true senza crashare
- [ ] **Turni vuoti**: se non riceviamo ingredienti, non crashare, fare il minimo
- [ ] **Timeout agente**: se l'agente impiega troppo, la fase cambia e le azioni falliscono
- [ ] **Fase sbagliata**: non chiamare prepare_dish fuori dalla serving phase

### Domande aperte
- [ ] max_steps=15 è abbastanza? Troppo?
- [ ] planning_interval=1 spreca token? Meglio 2-3?
- [ ] Serve un fallback se Regolo.ai è lento/down?

---

## 9. Token efficiency

**Status: DA FARE**

I giudici valutano "token usage efficiency". Con Regolo gratis non paghiamo, ma meno token = agente più veloce = più azioni per turno.

### Ottimizzazioni
- [ ] **Prompt compatto**: non mandare tutte 287 ricette, solo le gold recipes
- [ ] **Poche chiamate GET**: cachare i dati, non richiedere ogni volta
- [ ] **planning_interval**: ogni quanti step serve un piano? 1 è troppo
- [ ] **Risposte concise**: nel prompt dire "be extremely concise"
- [ ] **Meno tool per fase**: speaking non serve get_market, serving non serve place_bid

### Idea: tool diversi per fase?
Dare all'agente solo i tool rilevanti per la fase corrente, così non spreca token a ragionare su tool inutili.

---

## 10. Datapizza Monitoring (punti bonus)

**Status: DA FARE — ultimi 2h**

Le rules dicono che usare il monitoring è valutato positivamente. Invitation code: `hackapizza2`.

### Da fare
- [ ] Registrarsi su datapizza-monitoring.datapizza.tech
- [ ] Integrare tracing nel codice (vedi docs datapizza-ai)
- [ ] Dashboard che mostra: step agente, tool calls, token usage, errori

---

## 11. Pitch / Presentazione

**Status: DA FARE — domenica mattina**

2 minuti micro-pitch, possibilmente 5 minuti se finalisti.

### Punti da raccontare
- Architettura agentica con ReAct loop
- Strategia "nicchia di lusso" (data-driven recipe selection)
- Memoria tra turni e apprendimento
- Gestione resiliente (reconnect, thread, fallback)
- Uso efficiente dei token
- Risultati: saldo, clienti serviti, reputazione

---

## Priorità di implementazione

| # | Cosa | Impatto | Effort | Quando |
|---|------|---------|--------|--------|
| 1 | Gold recipes nel prompt | ALTO | BASSO | Subito |
| 2 | Strategia bid adattiva | ALTO | MEDIO | Subito |
| 3 | Serving robusto | ALTO | MEDIO | Subito |
| 4 | Memoria tra turni | MEDIO | MEDIO | Dopo test |
| 5 | Rate limiting / resilienza | ALTO | BASSO | Dopo test |
| 6 | Token efficiency (tool per fase) | MEDIO | MEDIO | Dopo test |
| 7 | Market trading | MEDIO | BASSO | Se c'è tempo |
| 8 | Analisi competitor | BASSO | MEDIO | Se c'è tempo |
| 9 | Diplomazia | BASSO | BASSO | Se c'è tempo |
| 10 | Monitoring Datapizza | MEDIO | BASSO | Ultime 2h |
| 11 | Pitch | ALTO | MEDIO | Domenica mattina |
