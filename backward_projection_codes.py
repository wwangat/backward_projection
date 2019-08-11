#Written by Wei Wang (wwangat@gmail.com)

#For Projection Operator paper
#Method developed by <Wei Wang, Siqin Cao, Frederick Fu Kit Sheong, Xuhui Huang>

import numpy
from msmbuilder.msm import MarkovStateModel
from optparse import OptionParser
from scipy.linalg import eig
from numpy import matlib as mb

def renormalize_eigenvectors(right_eigenvectors, stationary_population, n_macrostate):
	#convert to the "right normalized convention" as used in our paper
	#As TPM is reversible, \phi and v satisfy \phi = numpy.linalg.inv(Dn)*v
	inv_Dn = numpy.linalg.inv(numpy.diag(stationary_population))
	for j in range(n_macrostate):
		right_eigenvectors[:, j] = right_eigenvectors[:, j]/numpy.sqrt(right_eigenvectors[:, j].T.dot(inv_Dn).dot(right_eigenvectors[:, j]))
	left_evs = inv_Dn.dot(right_eigenvectors) #check
	return left_evs, right_eigenvectors

def calculate_eigenvector_for_reconstructed_macroTPM(mapping, nMacro, nMicro, micro_stationary_population, micro_right_eigenvectors):
	#Now get best Vg. In "column convention"
	A = numpy.zeros([nMicro, nMacro])
	for j in range(nMicro):
		A[j, mapping[j]] = 1
	#Projection operator
	macro_stationary_population = A.T.dot(micro_stationary_population)
	inv_DN = numpy.linalg.inv(numpy.diag(macro_stationary_population))
	Proj_Operator = numpy.diag(micro_stationary_population).dot(A).dot(inv_DN).dot(A.T)
	Vg = numpy.zeros([nMacro, nMacro])
	Vg[:, 0] = A.T.dot(micro_right_eigenvectors[:, 0]) #make sure we normalize it
	Vg[:, 0] = Vg[:, 0]/sum(Vg[:, 0])
	for j in range(1, nMacro):
		temp_vector = A.T.dot(micro_right_eigenvectors[:, j])
		for k in range(j):
			temp_vector -= Vg[:, k].T.dot(inv_DN).dot(A.T).dot(micro_right_eigenvectors[:, j])*Vg[:,k] #a number
		#normalize this vector
		temp_vector = temp_vector/numpy.sqrt(temp_vector.T.dot(inv_DN).dot(temp_vector))
		Vg[:, j] = temp_vector ###
	phi_g = inv_DN.dot(Vg)
	return Vg, phi_g, A #normalized right, left eigenvectors, membership matrix. column normalized convention

def evaluate_Ymatrix(mapping, nMicro, nMacro, micro_stationary_population, micro_right_eigenvectors):
	[microstate_left_eigenvector, micro_right_eigenvectors] = renormalize_eigenvectors(micro_right_eigenvectors, micro_stationary_population, nMacro)
	[right_Mg, left_Mg, membership] = calculate_eigenvector_for_reconstructed_macroTPM(mapping, nMacro, nMicro, micro_stationary_population, micro_right_eigenvectors)
	Y_matrix = numpy.zeros([nMacro, nMacro])
	for j in range(nMacro):
		for k in range(nMacro):
			Y_matrix[j, k] = abs(left_Mg[:, j].T.dot(membership.T).dot(micro_right_eigenvectors[:, k])/numpy.sqrt((left_Mg[:, j].T.dot(right_Mg[:, j]))*(microstate_left_eigenvector[:, k].T).dot(micro_right_eigenvectors[:, k])))
	return Y_matrix, left_Mg, right_Mg

def reconstruct_matrix(Vg, phi_g, microstate_eigenvalues, nMacro):
	print("return column normalized reconstructed macrostate TPM")
	return Vg.dot(numpy.diag(microstate_eigenvalues[:nMacro])).dot(phi_g.T) #column normalized matrix

def MFPT_using_Mg(TPM, lagtime): #return MFPT with column convention
	#check whether TPM is row-normalized or column-normalized
	n_state = TPM.shape[0]
	if(sum(abs(numpy.sum(TPM, axis=1)-numpy.ones([n_state])))<1e-10):
		print("row normalized matrix")
	elif(sum(abs(numpy.sum(TPM, axis=0)-numpy.ones([n_state])))<1e-10):
		print("column normalized matrix, converting to row first")
		TPM = TPM.T
	else:
		print("you should input a normalized TPM")
		exit()
	[eig_value, left_evs] = eig(TPM, right=False, left=True)
	sort_order = numpy.argsort(numpy.real(eig_value))[::-1] #in descending order
	left_evs = left_evs[:, sort_order]
	stat_pop = left_evs[:, 0]/numpy.sum(left_evs[:, 0])
	W_matrix = mb.repmat(stat_pop.T, n_state, 1)
	M_fund = numpy.linalg.inv(numpy.eye(n_state)-TPM+W_matrix)
	MFPT = numpy.zeros([n_state, n_state])
	for j in range(n_state):
		for k in range(n_state):
			MFPT[j, k] = (M_fund[k, k]-M_fund[j, k])/stat_pop[k]
	MFPT = MFPT.T*lagtime #convert to column convention
	return MFPT

#####################################################################################

parser = OptionParser()

parser.add_option('-l',"--trajlist_file",help="the trajlist indicating all the microstate assignment")
parser.add_option('-m',"--mapping_file",help="the microstate to macrostate mapping relationship, start from 0")
parser.add_option('-t','--lag_time_msm',help='lagtime is needed',type='int')
parser.add_option('-x', '--microstate_TPM', help='microstate transition probability matrix in row convention')
parser.add_option('-o', '--output_filename_prefix', help='prefix for the output')


(options, args) = parser.parse_args()

if(options.mapping_file is None):
	print("please input mapping file")
else:
	mapping = numpy.loadtxt(options.mapping_file, dtype='int')
	mapping = mapping-numpy.min(mapping) #now starts from 0
	nMacro = numpy.max(mapping)-numpy.min(mapping)+1 #number of macrostates
	nMicro = len(mapping) #number of microstates

if options.microstate_TPM is None:
	if options.trajlist_file is None and options.lag_time_msm is None:
		print("please input either the microstate TPM or the assignment list; follow instruction given by python backward_projection_codes.py -h")
		exit()
	else: #read in the trajlists and build MSM
		#read in assignments
		assignment_array = []
		for line in open(options.trajlist_file):
			assignment_array.append(numpy.loadtxt(line.strip()))
		msm_model = MarkovStateModel(lag_time = options.lag_time_msm, reversible_type='transpose', sliding_window=1, ergodic_cutoff='on', verbose=True, n_timescales = nMacro)
		msm_model.fit(assignment_array)
		microstate_TPM = msm_model.transmat_ #row normalized TPM
		microstate_TPM = microstate_TPM.T #column normaliezd convention
		microstate_right_eigenvector = msm_model.left_eigenvectors_[:, :nMacro] #only keep the top nMacro modes
		microstate_eigenvalues = msm_model.eigenvalues_[:nMacro] #only keep the top nMacro eigenvalues
		microstate_population = microstate_right_eigenvector[:, 0]/microstate_right_eigenvector[:, 0].sum()
else: #we already have the microstate TPM
	microstate_TPM = numpy.loadtxt(options.microstate_TPM) #row normalized 
	#countmatrix
	if(sum(abs(numpy.sum(microstate_TPM, axis=1)-numpy.ones([nMicro])))<1e-10):
		print("row normalized matrix, converting to column normalized first")
		microstate_TPM = microstate_TPM.T
	elif(sum(abs(numpy.sum(microstate_TPM, axis=1)-numpy.ones([nMicro])))<1e-10):
		print("column normalized matrix, satisfy our convention")
	else:
		print("you should input a row- or column-normalized TPM")
		eixt()
	[microstate_eigenvalues, microstate_right_eigenvector] = eig(microstate_TPM, left=False, right=True) 
	#sort the eigenvectors according to the eigenvalues (in descending order)
	microstate_eigenvalues = numpy.real(microstate_eigenvalues)
	sort_order = numpy.argsort(microstate_eigenvalues)[::-1] #in descending order
	microstate_eigenvalues = microstate_eigenvalues[sort_order]
	microstate_right_eigenvector = microstate_right_eigenvector[:, sort_order]
	microstate_eigenvalues = microstate_eigenvalues[:nMacro]
	microstate_right_eigenvector = microstate_right_eigenvector[:, :nMacro] #top modes, which corresponds to the top nMacro right eigenvectors in the column normalized convention for TPM (which we applied in the paper)
	microstate_population = microstate_right_eigenvector[:, 0]/microstate_right_eigenvector[:, 0].sum()

#Evaluate quality of lumping
#get the macrostate eigenvectors (of the reconstructed matrix)
[Y_matrix, left_Mg, right_Mg] = evaluate_Ymatrix(mapping, nMicro, nMacro, microstate_population, microstate_right_eigenvector)

print("Y_matrix in column convention\n", Y_matrix)

Mg = reconstruct_matrix(right_Mg, left_Mg, microstate_eigenvalues, nMacro)

print("Mg in column convention:\n", Mg)

MFPT_result = MFPT_using_Mg(Mg, options.lag_time_msm)

print("MFPT in column convention, calculated based on Mg:\n", MFPT_result)

print('writing the outputs')
numpy.savetxt(options.output_filename_prefix+'.Y_matrix', Y_matrix)
numpy.savetxt(options.output_filename_prefix+'_Mg_colnormalized.dat', Mg)
numpy.savetxt(options.output_filename_prefix+'_MFPT_column_convention.dat', MFPT_result)






