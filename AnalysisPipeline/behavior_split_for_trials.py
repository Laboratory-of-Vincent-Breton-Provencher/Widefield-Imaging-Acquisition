# utiliser create list trial dans prep data



# AP_times = np.load(r"AnalysisPipeline\Air_puff_timestamps.npy")

attente = 30
stim = 5 #int(input("Duration of opto stim(to create adequate timestamps)"))
Ns_aft = 15 #int(input("Seconds to analyze after onset of opto stim (trying to gte back to baseline)"))
opto_stims = np.arange(attente, 1000, attente+stim)