# Propagazione dei Raggi Cosmici

## Progetto di Metodi Computazionali per la Fisica

Viene implementata una simulazione della propagazione dei raggi cosmici nella galassia, tramite metodo random walk 3D. La simulazione è finalizzata nel ricavare la proprietà di isotropia osservata nei raggi cosmici e dovuta alla diffusione delle particelle nella galassia che subiscono fenomeni di scattering da parte del campo magntico galattico.

---
### CONTENUTO REPOSITORY:

**Propagazione_RC.pdf**: 

File che descrive la teoria utilizzata per la simulazione, si troverà anche una spiegazione della logica di quest'ultima con una descrizione anche dell'analisi dei dati che viene svolta e dei risultati ricavati.

**Bibliografia**: 

Cartella con i file .pdf utilizzati per ricavare i parametri e da cui è tratta la teoria alla base della simulazione. Nello specifico si trovano i seguenti testi:

   * Longair, High Energy Astrophysics, third edition

   * Gould, An Introduction to Computer Simulation Methods, first edition

   * Tomassetti, Astrofisica dei Raggi Cosmici: Fenomenologia e Modelli

**Simulazione**: 

Cartella contenti i codici e dati utilizzati per la simulazione, nello specifico si ha:

   * sim_data.csv: file che contiene i dati della simulazione
	
   * propagation_RC.py: file Python dove è presente la simulazione della propagazione
	
   * run_parallel.py: file con script Python che permette l'esecuzione in parallelo della simulazione
	
---							 
### ISTRUZIONI PER SCRIPT E FILE CSV:
							 
* **propagation_RC.py**

Contiene la simulazione della propagazione dei raggi cosmici. 
Per maggiori informazioni sulla fisica, la teoria e l'analisi dati consultare il pdf 'Propagazione_RC'.

Lo script può essere lanciato con i seguenti argomenti

 * -p PAR: Lancia la simulazione con un numero di particelle definito dall'utente, (Default=1000)
 
 * -t : La simulazione viene eseguita per sole 5 particelle e viene mostrato il grafico delle traiettorie 
 * -np : Viene stampato sul terminale il numero di particelle rilevate contenuto nel file .csv (Non viene eseguita la simulazione)
 
 * -s : Permette di salvare i dati delle particelle rilevate della simulazione, se non scelto i dati non verranno scritti
 
 * -d : Esegue l'analisi dati a partire da quelli presenti nel file .csv, stampa sul terminale i risultati (Non viene eseguita la simulazione)
 
Durante l'esecuzione della simulazione una barra di avanzamento dello stato terrà conto dello stato di completamento della particella, indicando il numero di particelle simulate al secondo e stimando il tempo di completamento della simulazione. 
Una volta terminato il processo vengono stampate sullo schermo le informazioni delle particelle rilevate decadute o fuggite dalla galassia. Insieme a queste informazioni, se scelto, verrà mostrato il grafico delle traiettorie e il nome del file in cui sono state salvate le particelle rilevate. 

* **sim_data.csv** 

Contiene i dati di direzione e tempo di arrivo delle particelle rilevate, organizzati in 4 colonne denominate:

 * 'Time'
 * 'dir_x' 
 * 'dir_y'
 * 'dir_z' 

Ogni volta che si esegue la simulazione con argomento '-s' i dati delle particelle rilevate vegnono sovrascritti e salvati nel file. 

Il file contiene 10072 particelle rilevate.	

* **run_parallel.py**

Contiene lo script che esegue in parallelo 4 simulazioni su 4 core logici della CPU tramite concurrent.futures e subprocces. 

Subprocess permette di eseguire lo script 'propagation_RC.py' con argomento già impostato '-s' in modo tale da salvare i dati sul file apposito. 

Grazie a questo script si può avere una presa dati più veloce. 

Ogni esecuzione occupa in media il 20% della RAM e il 10% della CPU. (INTEL i7, 8°gen e 8gb RAM)

Viene mostrato l'orario in cui il programma è stato eseguito e la fine del processo stimata, questa si aggira intorono ai 13 minuti in media. Al termine dell'esecuzione dei processi vengono stampati sullo schermo i risultati della simulazione per ogni simulazione come in 'propagation_RC.py' e il tempo di esecuzione dello script. 

---
### LIBRERIE UTILIZZATE

* Os
* Subprocess
* Concurrent.futures
* Time
* Datetime
* Numpy
* Pandas
* Matplotlib.plotly
* Argparse
* Tqdm

---


	
	
