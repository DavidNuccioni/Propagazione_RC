import sys, os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def parser_arguments():
	"""
	Funzione che definisce gli argomenti da passare quando si esegue il codice
	"""
	
	parser = argparse.ArgumentParser(description='Simulazione della propagazione dei raggi cosmici nella galassia')
	parser.add_argument('-p', '--par', type=int, action='store', default=1000, help='Inserisci il numero di particelle (Default: 1000)')
	parser.add_argument('-s', '--save', action='store_true', help='Salva i dati della simulazione')
	parser.add_argument('-d', '--data', action='store_true', help='Esegue analisi dei dati')
	parser.add_argument('-t', '--traj', action='store_true', help='Stampa le traiettorie di 5 particelle')
	parser.add_argument('-np', '--Npart', action='store_true', help='Visaulizza il numero di particelle già simulate e salvate nel file per analisi dati')
	
	return  parser.parse_args()


def save_data(arrivals_time, arrivals_dir):	
	"""
	Crea un dataframe con i dati della simulazione e lo salva nel file sim_data.csv
	"""
	
	# Creazione dataframe vuoto
	df = pd.DataFrame(data={ 
	'Time'	: arrivals_time,
	'dir_x'	: [arrivals_dir[:,0]],
	'dir_y'	: [arrivals_dir[:,1]],
	'dir_z'	: [arrivals_dir[:,2]],
	})
	
	# Salvataggio del dataframe in file .csv
	csvname = 'sim_data.csv'
	file_exists = os.path.exists(csvname)
	df.to_csv(csvname, mode='a' if file_exists else 'w', index=False, header=not file_exists)
	
	return csvname


def csv_df():
	"""
	Legge il file dei dati, se esiste, e lo trasforma in un dataframe per fare analisi dati
	"""
	
	df = None
	
	try:
		df = pd.read_csv('sim_data.csv')
	except FileNotFoundError:
		print("\n-------------------------------------------------------------------")
		print("Il file non è stato trovato, assicurati che esista")
		print("Eseguire la simulazione con salvataggio dei dati per creare il file")
		print("-------------------------------------------------------------------\n")
	
	return df
	

def data_analysis():
	"""
	Calcola il coefficiente di isotropia e il tempo medio di diffusione
	"""
	
	# Valori attesi per tempo di diffusione [ys] e coefficiente isotropia
	t_diff = 40000
	A_obs = 10e-4
	
	# Lettura dati dal dataframe e conversione in array
	df = csv_df()
	dir_ar = df[["dir_x", "dir_y", "dir_z"]].to_numpy()
	time_ar = df["Time"].to_numpy()

	# Calcola media dei tempi e deviazione standard
	N = len(time_ar)
	t_mean = np.mean(time_ar) / sec_y
	t_std = np.std(time_ar, ddof=1) / sec_y
	stderr_t = t_std / np.sqrt(N)
	
	# Test statistico rispetto al risultato atteso
	z_t = (t_mean - t_diff) / stderr_t

	# Calcola coefficiente anisotropia e deviazione standard
	d_mean = dir_ar.mean(axis=0)
	d_std = np.std(dir_ar, axis=0, ddof=1)
	stderr_d = d_std / np.sqrt(N)
	A = np.linalg.norm(d_mean)
	sigma_A = np.sqrt(np.sum((d_mean/A)**2 * stderr_d**2))
	
	# Test statistico rispetto al risultato atteso
	z_A = (A-A_obs) / sigma_A
	
	# Stampa i risultati
	print(f"\n------------------------------")
	print(f"Ipotesi diffusività:")
	print(f"------------------------------")
	print(f"Tempo medio = {int(t_mean)}ys") 
	print(f"Devst della media = {int(stderr_t)}ys")
	print(f"Test z = {z_t:.2f}")
	print(f"------------------------------")
	print(f"Ipotesi isotropia:")
	print(f"------------------------------")
	print(f"Coefficiente A medio = {A:.4f}")
	print(f"Deviazione std = {sigma_A:.6f}")
	print(f"Test z = {z_A:.2f}")
	print(f"------------------------------\n")
	
	return 

	
def plot_traj(position):
	"""
	Fornisce il plot delle traiettorie
	"""
	
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')
	
	# Plot delle traiettorie
	c = ['limegreen', 'olivedrab', 'chocolate', 'royalblue', 'darkorchid']
	for i, traj in enumerate(position):
		if len(traj) == 0:
			continue
		xs = [p[0]/pc for p in traj]
		ys = [p[1]/pc for p in traj]
		zs = [p[2]/pc for p in traj]
		ax.plot(xs, ys, zs, color=c[i], label=f'Particella {i+1}')
	
	# Sorgente con alone
	ax.scatter(0.0, 0.0, 0.0, color='lightcoral', s=1000, alpha=0.5, edgecolors='none')
	ax.scatter(0.0, 0.0, 0.0, c='red', s=50, label='sorgente')
	
	# Rilevatore con alone
	ax.scatter(x_ril/pc, y_ril/pc, z_ril/pc, color='cyan', s=1000, alpha=0.5, edgecolors='none')
	ax.scatter(x_ril/pc, y_ril/pc, z_ril/pc, color='blue', s=50, label='rilevatore')
	
	# Confine galassia
	z = np.linspace(-L_g/pc, L_g/pc, 80)
	theta = np.linspace(0, 2*np.pi, 80)
	theta_grid, z_grid = np.meshgrid(theta, z)
	x_cyl = R_d/pc * np.cos(theta_grid)
	y_cyl = R_d/pc * np.sin(theta_grid)
	z_cyl = z_grid
	ax.plot_surface(x_cyl, y_cyl, z_cyl, rstride=10, cstride=10, color='lightgrey', alpha=0.1, edgecolor='none')
	
	# Caratterizzazione del grafico
	ax.tick_params(axis='both', which='major', labelsize=14)
	ax.set_xlabel("x [pc]")
	ax.set_ylabel("y [pc]")
	ax.set_zlabel("z [pc]")
	ax.set_title("Grafico delle traiettorie di 5 particelle")
	ax.legend()
	plt.tight_layout()
	plt.show()
	
	return


def scatter_particle():
	"""
	Scattera la direzione della particella in una direzione casuale 
	"""	
	
	# Definizioni dei valori che vengono generati casualmente per phi e theta
	cos_th = np.random.uniform(-1.0, 1.0)
	sin_th = np.sqrt(1-cos_th**2)
	phi = np.random.uniform(0.0, 2.0*np.pi)
	
	# Definizione dei versori
	dir_x = sin_th * np.cos(phi)
	dir_y = sin_th * np.sin(phi)
	dir_z = cos_th
	rand_dir = [dir_x, dir_y, dir_z]
	
	return rand_dir
	
	
def simulation():
	"""
	Funzione principale che implementa la propagazione
	"""
	
	# Definizione dei parametri e delle variabili iniziali
	#---------------------------------------------------------------------
	N_par = args.par					# Numero di particelle che vengono simulate
	N_ril = 0							# Numero di particelle rilevate
	N_exit = 0							# Numero di particelle fuggite
	N_dec = 0							# Numero di particelle decadute
	arrivals_dir = []					# Direzione di arrivo 
	arrivals_time = []					# Tempo di arrivo

	if args.traj:
		N_par = 5						# Numero di particelle per il grafico
		position = [[(0.0,0.0,0.0)] for i in range(N_par)]	# Lista per triaettorie
	#---------------------------------------------------------------------
	
	# Inizio della propagazione
	#---------------------------------------------------------------------
	print(f"\nCompletamento del processo per {N_par} particelle...")
	for p in tqdm(range(N_par)):
		
		# Creazione particella al centro della galssia con direzione iniziale casuale	
		x = 0.0							# Posizione iniziale x
		y = 0.0							# Posizione iniziale y
		z = 0.0							# Posizione iniziale z
		t_part = 0.0					# Tempo di vita iniziale
		
		for j in range(int(step_max)):
			
			# Randomizzazione direzione particella
			dir_i = scatter_particle()		# Array con componenti direzione casuale 
			
			# Movimento della particella 
			x_new = x + mfp * dir_i[0]
			y_new = y + mfp * dir_i[1]	
			z_new = z + mfp * dir_i[2]
			t_part += t_mfp
			
			# Salva la traiettoria
			if args.traj:
				position[p].append((x_new, y_new, z_new))	

			# Controllo fuga dalla galassia
			r_xy = np.sqrt(x_new*x_new + y_new*y_new)
			if (r_xy > R_d) or (abs(z_new) > L_g):
				N_exit += 1

				break

			# Controllo particelle decadute 
			if t_part >= t_max: 
				N_dec += 1 
				
				break 
			
			# Controllo particella rilevata nel Sistema Solare e salvataggio dati
			dx = x_new - x_ril
			dy = y_new - y_ril
			dz = z_new - z_ril
			
			if dx*dx + dy*dy + dz*dz <= R_ril * R_ril:
				arrivals_dir.append(np.array([dir_i[0], dir_i[1], dir_i[2]]))
				arrivals_time.append(t_part)
				N_ril += 1

				break

			# Aggiornamento variabili
			x = x_new 
			y = y_new 
			z = z_new	
	#---------------------------------------------------------------------
	
	 
	# Salva i dati di ogni particella rilevata se scelto come argomento
	#---------------------------------------------------------------------
	if args.save:
		save_data(arrivals_time, arrivals_dir)
	#---------------------------------------------------------------------
	
	# Stampa informazioni sulla simulazione
	#---------------------------------------------------------------------
	print(f"---------------------------")
	print('Particelle rilevate:', N_ril)
	print('Particelle fuggite:', N_exit)
	print('Particelle decadute:', N_dec)
	print(f"---------------------------")
	#---------------------------------------------------------------------
	
	# Se scelto, mostra il grafico delle traiettorie
	#---------------------------------------------------------------------
	if args.traj:
		for i in range(N_par):
			print(f"Distanza in parsec percorsa dalla particella {i+1}: {len(position[i])}")
		plot_traj(position)	
	#---------------------------------------------------------------------
	
		
if __name__=="__main__":
	
	# Definizione delle costanti
	#---------------------------------------------------------------------
	c     = 3e8							# Velocità particella (v=c)  [m/s]
	sec_y = 31536e3						# Secondi in un anno		 [s]
	pc    = 3.086e16					# Parsec in metri   		 [m]
	L_g   = 1500.0 * pc					# Semialtezza galassia 	     [m]
	R_d   = 3500.0 * pc					# Raggio disco galattico     [m]
	R_ril = 15.0 * pc					# Raggio del rilevatore      [m]
	x_ril = 0.0 * pc					# Coordinata x rilevatore    [m]
	y_ril = 0.0	* pc					# Coordinata y rilevatore    [m]
	z_ril = 200.0 * pc					# Coordinata z rilevatore    [m]
	mfp   = 10 * pc						# Cammino libero medio (1pc) [m]
	t_mfp = mfp/c						# Tempo cammino libero medio [s]
	t_max = 3e6 * sec_y					# Tempo max propagazione     [s]
	step_max = 1e5						# Passi max di propagazione  
	#---------------------------------------------------------------------
	
	# Richiamo della funzione degli argomenti
	#---------------------------------------------------------------------
	args = parser_arguments()
	#---------------------------------------------------------------------
	
	# Mostra il numero di particelle salvate nel file
	#---------------------------------------------------------------------
	if args.Npart:
		df = pd.read_csv('sim_data.csv')
		print(f"\n--------------------------------------------")
		print('Particelle rilevate e salvate nel file:', df.shape[0])
		print(f"--------------------------------------------")
	#---------------------------------------------------------------------
	
	# Esegue analisi dati 
	#---------------------------------------------------------------------
	elif args.data:
		data_analysis()
	#---------------------------------------------------------------------
	
	# Esegue la simulazione
	#---------------------------------------------------------------------
	else:
		simulation()
	#---------------------------------------------------------------------
	
