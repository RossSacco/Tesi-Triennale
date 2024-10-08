# Tesi di Laurea Triennale

# Design, sviluppo e ottimizzazione di una rete neurale convoluzionale (CNN) per il riconoscimento dei picchi del battito cardiaco dall'elettroencefalogramma

<p>L'obiettivo dello studio è quello di costruire un modello robusto in grado di generalizzare su dati eterogenei. Sono stati testati diversi iperparametri (batch_size, learning_rate, momentum) e ottimizzatori (SGD, ADAM) utilizzando tecniche di GridSearch Cross-Validation per ottenere configurazioni ottimali. Inoltre, sono stati implementati esperimenti con il metodo Leave-One-Subject-Out (LOSO) per migliorare le capacità di generalizzazione.</p>

## ABSTRACT

<P>
  Nell’ambito del Machine Learning il riconoscimento dell’attività umana, o Human Activity Recognition (di seguito HAR), è il problema di determinare l’azione che sta compiendo un essere umano. L’obiettivo dell’HAR è quello di identificare e classificare le attività umane da dati grezzi provenienti da sensori, come ad esempio accelerometri o giroscopi presenti in smartphone, smartwatch o dispositivi indossabili.

  
Tale problema è molto comune nell’ambito del Machine Learning, essendo applicabile in diversi contesti e applicazioni, tra cui il monitoraggio della salute, il fitness tracking, la sicurezza e la domotica. Alcuni esempi di attività umane che possono essere riconosciute tramite l’analisi dei dati in ques@one includono camminare, correre, salire le scale, guidare, sedersi, dormire e molti altri. Questo riconoscimento presenta tuttavia diverse sfide e problema@che a cui prestare attenzione nell’ambito del Machine Learning per poter ottenere dei risultati affidabili ed ottimali. Essendo i soggetti da cui provengono i dati degli esseri umani, per natura molto eterogenei in termini di velocità, resistenza e altre caratteristiche, può risultare difficile costruire un modello che riesca a generalizzare concretamente i dati su soggetti così differenti. Altre problematiche potrebbero essere i fattori ambientali che influenzano l’attvità, la grande dimensionalità dei dati in esame e l’overfitting.


Lo studio condotto si concentra nell’analisi di dati raccolti tramite EleGroencefalogramma (EEG) su un insieme di 26 soggetti. Si utilizza una Convolutional Neural Network (anche CNN) Unidimensionale (o 1D) come modello per la classificazione e individuazione di picchi del battito cardiaco. Questa tipologia di modello permette di ottenere risultati all’avanguardia su compiti impegnativi come quello in esame, utilizzando funzionalità d’apprendimento su dati grezzi, senza quindi il bisogno di un intervento ingegneristico per l’utilizzo di dati manipolati. Il lavoro, inizialmente si è concentrato sulla valutazione delle prestazioni del modello utilizzando diverse configurazioni di iperparametri nell’architettura e nell’addestramento, utilizzando come metrica di valutazione l’accuratezza, con lo scopo di individuare i valori che fornissero risultati ottimali.


Nello specifico, sono state testati e individuati i valori ottmali degli iperparametri “batch_size”, “learning_rate”, “momentum” ed è stato individuato l’ottmizzatore più adeguato, considerando lo Stochastic Gradient Descendent (anche SGD) e ADAM. Per farlo, è stata utilizzata la tecnica di GridSearch Cross-Valida0on, fornita dalla libreria di Python ‘scikit-learn’, con la quale è stato possibile effeGuare una ricerca incrociata. Successivamente, sono stati testati diversi settings sperimentali che permettessero di verificare quale fosse il miglior modo di aumentare le capacità di generalizzazione del modello. Nello specifico è stato testato il se1ng sperimentale Leave-One-Subject-Out (anche LOSO) e successivo affinamento dei risultati tramite tecnica di Fine-Tuning con diverse configurazioni.

I risultati ottenuti sono stati utili a identificare quali siano i migliori settaggi di un modello sequenziale affinché possa classificare adeguatamente i dati forniti con grandi capacità di generalizzazione.
</P>

## Lo studio completo è riportato nella documentazione, insieme ad alcuni degli script utilizzati! ☝🏻
