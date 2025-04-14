import matplotlib.pyplot as plt
import numpy as np


path = r"C:\Users\gabri\Documents\Université\Maitrise\Projet\Widefield-Imaging-Acquisition\AnalysisPipeline\specs sys optique\QE_moment_10px.csv"
QE = np.loadtxt(path, delimiter=";").transpose()


plt.plot(QE[0], QE[1], color='k')
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Efficacité quantique (%)")
plt.xlim(QE[0].min(), QE[0].max())
# plt.vlines((405, 470, 530, 625, 785), QE[1].min(), QE[1].max())
plt.savefig("moment_QE.png", dpi=600)
plt.show()

# 405: 55%
# 470: 70%
# 530: 73%
# 625: 65%
# 785: 32%