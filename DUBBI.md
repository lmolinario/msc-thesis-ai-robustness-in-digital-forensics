Ciao Silvia, mentre leggi i primi tre capitoli ho fatto qualche piccola aggiunta: nel capitolo di background ho inserito la parte matematica e descrittiva degli attacchi adversarial e anti-forensic che sto usando.

Nel frattempo sto buttando giù il Capitolo 4 sulla metodologia e mi sono venuti alcuni dubbi. I principali sono questi:

1. La selezione finale del dataset è stata fatta con un protocollo human-in-the-loop, ma da un singolo revisore. Ho già indicato questo aspetto come limite metodologico, specificando che i criteri erano espliciti, che il dataset è frozen e che tutto è tracciato tramite manifest e log. Secondo te è sufficiente per una tesi, oppure sarebbe meglio prevedere almeno una seconda revisione su un sottoinsieme del dataset? Il dubbio mi viene dal fatto che, da quanto ho visto, la letteratura sull’accordo inter-annotatore considera metodologicamente più forte l’uso di più annotatori o revisori indipendenti.

2. La classe OOD l’ho descritta come un insieme di casi borderline o fuori distribuzione, usati per stress-testare modelli e tool. In pratica vorrei valutare se questi sistemi tendono a forzare immagini anomale, ambigue o fuori dominio dentro una delle due classi operative, weapon o non_weapon, magari anche con alta confidenza. Vorrei capire se questa impostazione è corretta o se, secondo te, va resa ancora più prudente.

3. Nella classe non_weapon c’è uno sbilanciamento per sorgente, perché molte immagini provengono dalla fonte deepweb. Ho esplicitato il limite, chiarendo che il dataset è bilanciato per classe ma non perfettamente per sorgente. Secondo te basta dichiararlo così? Naturalmente, nel bundle dato in pasto ai tool non c’è alcun riferimento esplicito alla sorgente, perché viene usato un blind bundle.

4. Per gli strumenti forensi sto usando una formulazione prudente: tratto i tool con moduli AI/proprietari come black-box o quasi black-box. Non so però se sia il caso di prevedere anche Autopsy come baseline o ambiente di supporto per ingest, hashing, tagging e reporting, senza presentarlo come equivalente diretto ai tool commerciali con moduli AI. Anche qui vorrei capire se il perimetro metodologico è corretto.

In sostanza, vorrei evitare di presentare il protocollo come più “forte” di quanto sia, ma allo stesso tempo renderlo difendibile dal punto di vista metodologico.

Dal punto di vista dei test reali, invece, sono pronto: se dai un’occhiata alla repo, mi sembra abbastanza allineata ed è già pronto il forensic_evaluation_bundle con tutti gli 11.500 file da testare. I file sono importabili in modalità blind e sono tracciati esternamente tramite manifest e hash, così da poterli ricondurre successivamente alle rispettive classi di appartenenza.

Quindi potrei iniziare a estrarre i dati sperimentali, ma prima vorrei allineare bene tutta la parte metodologica, in modo che sia coerente, prudente e accademicamente difendibile.
