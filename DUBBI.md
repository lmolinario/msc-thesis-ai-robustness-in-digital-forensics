parametri attacchi adv e settining antiforensic



Ciao Lello! Sono riuscita a finire di leggere la tua tesi. ti mando alcuni commenti. mi ricordi quando vorresti laurearti?

Titolo: Evaluating the Robustness of AI-based Forensics Tools under Adversarial and Anti-Forensics Attacks **(ho cambiato titiolo mi sembrava piú leggibile...)**

Prima di 1.1 metterei in breve perché nasce questa tesi e obiettivo e focus principale. Un mini paragrafo (da fare)

Nel paragrafo 1.1 gli acronimi vanno prima scritti per intero la prima volta poi usati sempre come acronimi.  **OK**
Esempio Digital Forensics (DF) la prima volta poi sempre DF **(mettiamo acronimi sempre  in Grassetto??)**

Nel paragrafo 1.2 ti suggerisco di parlare anche dello shock dell’analista nel vedere certe immagini 
ripetutamente https://www.sciencedirect.com/science/article/pii/S1742287619301549?via%3Dihub **OK**

Nel paragrafo 1.3 approfondirei leggermente (perché poi spiegato bene nel background) le tecniche di anti-forensics a
prescindere da AI quindi tutte quelle che consentono di non rilevare qualcosa sia all’analista umano sia ai tool 
(anche senza AI) esempio steganografia **OK**

Nel paragrafo 1.4 parli del problema del dataset. Qui “Ciò implica consider-
are dataset realistici, contenuti eterogenei, immagini soggette a trasformazioni plausibili, casi
borderline e situazioni in cui la distinzione tra contenuto rilevante, irrilevante o ambiguo non è sempre netta“ 
citerei qualche lavoro (DFRWS ne ha diversi) **(Ok ne ho trovati due piú pertinenti silva2021microservices per 
classificazione AI di immagini da evidenze criminali, incluse armi e mckeown2024phaser
 per necessità di dataset e protocolli valutativi aderenti allo scenario operativo)**

Paragrafo 1.5 “Alla luce delle criticitá evidenziate nei paragrafi precedenti, questa tesi si propone di valutare
in modo sistematico la robustezza operativa di strumenti e modelli basati su AI impiegati per
la classificazione automatica di immagini in ambito forense. L’obiettivo generale `e verificare
in quale misura tali sistemi mantengano prestazioni affidabili quando vengono esposti a input
manipolati, degradati o comunque non pienamente conformi alle condizioni ideali di addestra-
mento e validazione, con particolare attenzione a scenari in cui le perturbazioni possano essere
ricondotte a strategie adversariali o anti-forensi”. 
Qui alla fine della frase aggiungerei il fatto che queste manipolazioni possono essere fatte anche dai 
cybercriminali o criminali tradizionali appositamente per evadere i sistemi di rilevamento e non essere accusati. 
Tra l’altro ora con utilizzo di gen-AI è facile anche chiedere a un LLM anche general purpose come GPT come nascondere i file, 
farsi produrre del codice per nascondere 
(esempio mio paper ma ce ne sono molti altri sciencedirect.com/science/article/pii/S1742287619301549?via%3Dihub) 
e nel futuro non si esclude che possano generare automaticamente già l’immagine modificata. **(OK)**

1.5 Cosa vuol dire pipeline accademica? **(OK)**
Lo usi spesso questo termine accademico ma io non l’ho mai visto negli articoli scientifici**(OK)**

| Espressione                                                  | spiegazione                                                                                                   |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| pipeline accademica                                                         | valutazione controllata dei modelli proxy                                                                                   |
| pipeline operativo-forense                                                  | valutazione operativa black-box dei software AI-based selezionati                                                           |
| integrazione di una pipeline accademica e di una pipeline operativo-forense | integrazione tra la valutazione controllata dei modelli proxy e la valutazione operativa black-box dei software selezionati |

1.5 dataset OOD, anche qui spiegare significato acronimo **(OK Acronimo espanso come prima ricorrerza)**

Nel paragrafo 1.7 quando scrivi “Il problema assume rilievo particolare quando i sistemi automatici operano 
in modalità sostanzialmente opaca” qui citerei anche i lavori di explainable AI. Anche perché poi si ritrova 
dal nulla nel capitolo 2 e serve flusso e coerenza **(OK)**

1.8 più corto, di solito si scrive in una frase per capitolo **(OK)**

AML? Non ho mai visto adversarial machine learning abbreviato così e non puoi introdurlo così nella tesi (scusa) **(Adv-ML?? Come fa Nowroozi? nel frattempo lo tolgo dappertutto )**



Nel capitolo 2 manca discutere delle fasi forensi e soprattutto delle principali tecniche di acquisizione e analisi **(OK)**

Nel 2.4 parlerei anche di riconoscimento di oggetti che è proprio quello che fanno (suppongo) gli algoritmi di AI nei tool 
forensi perché riconoscono appunto l’oggetto esempio arma dentro immagine. **(OK)**

2.5.1 dici nell’ultimo paragrafo che “la definizione del threat model consente infine di collegare gli attacchi adversariali 
alle trasformazioni anti-forensi” e si capisce che un attacco antiforense può essere fatto solo se legato ad attacco adversarial
ma così non è. Infatti poi nel 2.6 lo spieghi bene

Nel 2.6 cosa intendi con “nel contesto dell’image forensics molte analisi si badano … su caratteristiche del sensore”. Che sensore? **(hai ragione non ho spiegato bene mi riferivo al (PRNU)  CCD o CMOS vedi 2.3)**



Paragrafo 3.1 quando dici “in terzo luogo, la validazione dei sistemi …” lì come citazione ci vanno 9 e 10 non 9 e 11

Nel 3.2 quando parli degli strumenti forensi, io farei una sezione a parte dove li descrivi ciascuno ed evidenzi 
caratteristiche e limiti. Ricorda che nella parte di background è importante anche parlare delle fasi forensi e 
metodologie di acquisizione e analisi, altrimenti chi non conosce la materia non può capire bene il contributo e lo studio

Nel 3.7 quando dici “ne consegue che la xai, nel contesto forense, deve essere intesa come supporto…” cita il 
lavoro https://arxiv.org/abs/2510.14638 altrimenti sembra una tua affermazione e a meno che nella parte dei risultati o della 
metodologia etc non devi fare affermazioni tue

3.8 quindi stai creando un framework? Non sono sicura che sia un framework, a meno che mi sia persa qualcosa, qualche contributo 
principale della tua tesi **(protocollo sperimentale adottato nella tesi.... c'é ne vuole a farlo diventare un Framework...)**

Figura a.1 NON si mette il QR code, metti direttamente il link…

Mi sfugge utilità a6

Comunque se mi mandi link di overleaf divento io proprietario così non c’è limite compilazione e i commenti si mettono direttamente lì senza questi passaggi

