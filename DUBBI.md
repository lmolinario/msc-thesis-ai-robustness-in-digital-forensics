I dubbi principali riguardano questi punti:

1. la selezione finale del dataset è stata fatta con protocollo human-in-the-loop, 
ma da un singolo revisore. Ho già indicato questo aspetto come limite metodologico, 
specificando che i criteri erano espliciti, il dataset è frozen e tutto è tracciato tramite manifest/log. 
Secondo lei è sufficiente per una tesi magistrale o sarebbe meglio prevedere una seconda revisione parziale
su un campione?

2. la classe OOD l’ho descritta non come una vera terza classe semantica omogenea, ma come insieme di casi
borderline/fuori distribuzione per stress testare i modelli e i tool. Vorrei capire se questa impostazione è
corretta o se va resa ancora più prudente.

3. nella classe non_weapon c’è uno sbilanciamento per sorgente, perché molte immagini provengono dalla fonte
deepweb. Ho esplicitato il limite, chiarendo che il dataset è bilanciato per classe ma non perfettamente per
sorgente. Secondo lei basta dichiararlo così?

4. per gli strumenti forensi sto usando una formulazione prudente: tratto i tool con moduli AI/proprietari come
black-box o quasi black-box, mentre strumenti come Autopsy li descrivo più come baseline/ambiente di supporto per
ingest, hashing, tagging e reporting. Anche qui vorrei capire se il perimetro è corretto.

In sostanza, vorrei evitare di presentare il protocollo come più “forte” di quanto sia, ma allo stesso tempo renderlo difendibile dal punto di vista metodologico. Secondo lei questi limiti, così formulati, sono accettabili o conviene rafforzare qualche punto?
