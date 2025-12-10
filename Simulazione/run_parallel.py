import sys, os
import subprocess
import concurrent.futures
import time
from datetime import datetime, timedelta


def run_simulation_block(execution_id, script_target):
    """
    Esegue lo script target per un singolo blocco di lavoro.
    """
    
    print(f"Avvio processo numero {execution_id}", flush=True)

	# Esegue lo script_target e ne salva l'output
    command = ['python3', script_target, '-s']
    result = subprocess.run(command, check=True, capture_output=True, text=True)

	# Stampa l'output dello script_target
    print(f"Esecuzione processo {execution_id} completata con successo.", flush=True)
    stri = f"Output processo {execution_id}: {result.stdout[50:300]}"
    
    return stri

	
def main():
	
	# Stampa informazioni su tempo di esecuzione
	now = datetime.now()
	stop = datetime.now() + timedelta(minutes=14)
	print(f"\nEsecuzione in parallelo di 4 processi {script_target} con 1000 particelle ciascuno\n")
	print(f"----------------------------------------")
	print(f"Inizio dell'esecuzione: {now.strftime("%H:%M")}")
	print(f"Termine dell'esecuzione stimato: {stop.strftime("%H:%M")}")
	print(f"----------------------------------------\n")
	
	# Numero totale di esecuzioni della simulazione
	n_exe = 4 
	
	# Inizializzazione del tempo e lista risultati
	start_time = time.time()
	results = []
	
	# Avvia le simulazioni in parallelo
	with concurrent.futures.ProcessPoolExecutor() as executor:
		futures = [executor.submit(run_simulation_block, i, script_target) for i in range(1, n_exe + 1)]
	
	# Salva i risultati delle simulazioni terminate nella lista
	for f in (concurrent.futures.as_completed(futures)):
			results.append(f.result())
	
	# Tempo trascorso
	end_time = time.time()
	tot_min = int((end_time - start_time)//60)
	tot_sec = int((end_time - start_time)%60)
	
	# Stampa dei risultati
	print("\nRiassunto di tutte le Esecuzioni \n ")
	for r in results:
		print(f"---------------------------")	
		print(r)
	print(f"-----------------------------------------------")
	print(f"Tempo totale trascorso: {tot_min:02d} minuti e {tot_sec:02d} secondi")
	print(f"Risultati salvati nel file: sim_data.csv")
	print(f"-----------------------------------------------")
	
if __name__=="__main__":

	# Nome script da eseguire
	script_target = './propagation_RC.py'

	main()

