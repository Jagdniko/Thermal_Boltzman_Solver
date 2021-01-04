import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, odeint, solve_ivp
from scipy.integrate import solve_ivp
from scipy.special import kv
from matplotlib import pyplot as plt
import warnings

class Boltzmann_solver():
    ''' This class solves Boltzmann equation for weakly coupled massless or effectively massless particles.
        Here Cp is the collision term, Cp is a function with first parameter is T_SM temperature of SM plasma, second parameter is T_NR.Please make sure yur collisoion terms works before input
        property is the thermal properties for the massless particle introduced, property is a tuple with
        [mass of the particle, effective degree of freedom, thermal type ("F for fermions "B" for bosons)]
    '''
    def __init__(self, Cp = None, property = [0,6,"F"], label = None):
        ''' results are restored in Boltzmann_solver.save variable including: thermal properties, label, solved time array,
                 solved density array array, solved temperature array, temperatrue relation by interpolation and initial temperature in solution
        '''
        self.save = []
        self.Cp = Cp
        self.Mass = property[0]
        self.Degf = property[1]
        self.Spin = property[2]
        self.label = label
        self.get_T_SM = _T_SM_rho
        self.rho_SM = lambda x: _rho_SM(x)
        self.p_SM = lambda x: _p_SM(x)
        self.get_T_NR = lambda x: _T_NR_rho(x,self.Mass,self.Spin,self.Degf)
        self.rho_NR = lambda x: Enrg(x,self.Mass,self.Spin,self.Degf)
        self.p_NR = lambda x: Pres(x,self.Mass,self.Spin,self.Degf)


    def drhodt(self, rho_vec, t):
        rhosm, rhonr = rho_vec
        if rhonr<0 or rhosm<0: raise ValueError("rho_nr ="+ str(rhonr) + "rho_sm ="+ str(rhosm) + "time = "+ str(t)  +"check your collision term")
        tsm = self.get_T_SM(rhosm)
        tnr = self.get_T_NR(rhonr)
        delta = FAC*self.Cp(tsm, tnr)
        d_rho_SMdt = (-_Hubble(rhosm + rhonr) * (3 * rhosm + self.p_SM(tsm)) - delta)
        d_rho_NRdt = (-_Hubble(rhosm + rhonr) * (3 * rhonr + self.p_NR(tnr)) + delta)
        return [d_rho_SMdt,d_rho_NRdt]

    def presolve_check_v1(self, T0, Tf, Gtol = 100, num = 1e2):
        #in this version we check that whether TE is reached from effective interaction strenth for NR particle when TE is reach,
        # since rho_NR < rho_TE_NR always holds therefore Gamma will be less than real Gamma. So this check will be even stricter than real situation .
        Tsm0 = T0[0]
        Tsample = np.logspace(np.log10(Tsm0),np.log10(Tf),num = np.log10(Tsm0/Tf)*num)
        Gamma = np.zeros_like(Tsample)
        for i, T in enumerate(Tsample):
            rho = self.rho_SM(T) + self.rho_NR(T)
            Gamma[i] = FAC*self.Cp(T,0)/(rho*_Hubble(rho))

        if np.all(Gamma<=Gtol): return T0
        else:
            Tstart = Tsample[Gamma>Gtol][-1]
            print("Effective interction strenth is larger than Gtol during solve, which indict thermal equilbrium is reached. Solve will begin at T =" + str(Tstart))
            warnings.warn("Effective interction strenth is larger than Gtol during solve, which indict thermal equilbrium is reached. Solve will begin at T =" + str(Tstart), DeprecationWarning)
            return [Tstart,Tstart]


    def solve(self, T0= [1e5, 1e2],Tf = 1e2, num = 10000, rtol = 1e-5, atol = 1e-5, mxstep = 5000, precheck = True):
        if self.Cp == None: raise ValueError
        if precheck == True:
            T0 = self.presolve_check_v1(T0, Tf)
        Tsm0, Tnr0 = T0
        rhosm0 = self.rho_SM(Tsm0)
        rhonr0 = self.rho_NR(Tnr0)
        rhosmf = self.rho_SM(Tf)

        ##roughly speaking: rho=t^(-1/2), +1 to ensure the solutions covers end Temperture
        time_folding = 1/2*np.log10(rhosm0/rhosmf) + 1
        t0 = 1. / (2 * _Hubble(rhosm0+rhonr0))
        solve_points_vec = np.logspace(np.log10(t0), np.log10(t0*10**(time_folding)), num=num)
        sol = odeint(self.drhodt, [rhosm0,rhonr0], solve_points_vec, rtol=rtol, atol=atol, mxstep=mxstep)
            # print(sol.shape)
        a = np.zeros_like(sol)
        for i in range(len(sol[:,0])):
            a[i, 0] = self.get_T_SM(sol[i, 0])
            a[i, 1] = self.get_T_NR(sol[i, 1])
        if a[-1,0] > Tf: raise ValueError("T final did not reached")
        TT = lambda x: np.exp(interp1d(np.log(a[:,0]),np.log(a[:,1]),bounds_error=False,kind='linear')(np.log(x)))
        record = {'Mass': self.Mass, 'Degree of freedom': self.Degf, 'Type': self.Spin, "Label": self.label,\
                  "time_solve" : solve_points_vec, "rho_solve" : sol, "T_solve" : a, "TT_curve": TT, "Tstart":T0[0]}
        self.save.append(record)
        return sol


# <editor-fold desc="Maxwell-Boltzman Thermaldynamics">
def Nmbr_MB(Temp, Mass, Degf):
    return Degf * Mass**2 * Temp / (2 * np.pi**2) * kv(2,Mass/Temp)
def Enrg_MB(Temp, Mass, Degf):
    return Degf * Mass**2 * Temp / (2 * np.pi**2) * (Mass*kv(1,Mass/Temp) + 3*Temp*kv(2,Mass/Temp))
def Pres_MB(Temp, Mass, Degf):
    return Degf * Mass**2 * Temp / (2 * np.pi**2) * Temp * kv(2,Mass/Temp)
def Sigm_MB(Temp, Mass, Degf):
    return Degf * Mass**2 * Temp / (2 * np.pi**2) * (Mass**2*kv(2,Mass/Temp) + 3*Temp*Mass*kv(3,Mass/Temp))
def dNmbrdT_MB(Temp, Mass, Degf):
    return Enrg_MB(Temp, Mass, Degf)/Temp**2
def dEnrgdT_MB(Temp, Mass, Degf):
    return Sigm_MB(Temp, Mass, Degf)/Temp**2
# </editor-fold>

# <editor-fold desc="Define Constants">
# Neglectable Constant or Cutting Number on e-folds
Cutn = 30.0

# All in MeV Units!
GF  = 1.1663787e-5*1e-6 #in MeV^{-2}
me  = 0.511
Mpl = 1.22091e19*1e3

# Conversion factor to convert MeV^-1 into seconds
FAC = 1./(6.58212e-22)

# sW2 =  1-Mw^2/Mz^2
sW2 = 0.223
# </editor-fold>

# <editor-fold desc="Thermaldynamics">
def Enrg(Temp, Mass, Spin, Degf):
    if Spin == "B":
        if Temp < Mass / Cutn:
            return Enrg_MB(Temp, Mass, Degf)
        elif Temp >= Mass * Cutn:
            return Degf * np.pi ** 2 / 30. * Temp ** 4
        else:
            return Degf / (2 * np.pi ** 2) * Temp ** 4 * \
                   quad(lambda E: E ** 2 * (E ** 2 - (Mass / Temp) ** 2) ** 0.5 / (np.exp(E) - 1.), Mass / Temp,
                        Mass / Temp + Cutn, epsabs=1e-12, epsrel=1e-12)[0]
    elif Spin == "F":
        if Temp < Mass / Cutn:
            return Enrg_MB(Temp, Mass, Degf)
        elif Temp >= Mass * Cutn:
            return Degf * 7. / 8. * np.pi ** 2 / 30. * Temp ** 4
        else:
            return Degf / (2 * np.pi ** 2) * Temp ** 4 * \
                   quad(lambda E: E ** 2 * (E ** 2 - (Mass / Temp) ** 2) ** 0.5 / (np.exp(E) + 1.), Mass / Temp,
                        Mass / Temp + Cutn, epsabs=1e-12, epsrel=1e-12)[0]
    else:
        raise ValueError

def Pres(Temp, Mass, Spin, Degf):
    if Spin == "B":
        if Temp < Mass/Cutn: return Pres_MB(Temp, Mass, Degf)
        elif Temp >= Mass*Cutn: return Degf * 1/3 * np.pi**2 / 30. * Temp**4
        else: return Degf/(6*np.pi**2)*Temp**4*quad(lambda E: (E**2-(Mass/Temp)**2)**1.5/(np.exp(E)-1.) ,Mass/Temp,Mass/Temp + Cutn,epsabs=1e-12,epsrel = 1e-12)[0]
    if Spin == "F":
        if Temp < Mass/Cutn: return Pres_MB(Temp, Mass, Degf)
        elif Temp >= Mass*Cutn: return Degf * 1/3 * 7./8. * np.pi**2/30. * Temp**4
        else: return Degf/(6*np.pi**2)*Temp**4*quad(lambda E: (E**2-(Mass/Temp)**2)**1.5/(np.exp(E)+1.) ,Mass/Temp,Mass/Temp + Cutn,epsabs=1e-12,epsrel = 1e-12)[0]

def dEnrgdT(Temp, Mass, Spin, Degf):
    if Spin == "B":
        if Temp < Mass/Cutn: return dEnrgdT_MB(Temp, Mass, Degf)
        elif Temp >= Mass*Cutn: return Degf * 4 * np.pi**2/30. * Temp**3
        else: return Degf/(2*np.pi**2)*Temp**3*quad(lambda E: 0.25*E**3*(E**2-(Mass/Temp)**2)**0.5*np.sinh(E/2.0)**-2 ,Mass/Temp,Mass/Temp + Cutn,epsabs=1e-12,epsrel = 1e-12)[0]
    if Spin == "F":
        if Temp < Mass/Cutn: return dEnrgdT_MB(Temp, Mass, Degf)
        elif Temp >= Mass*Cutn: return Degf * 7./8. * 4 * np.pi**2/30. * Temp**3
        else: return Degf/(2*np.pi**2)*Temp**3*quad(lambda E: 0.25*E**3*(E**2-(Mass/Temp)**2)**0.5*np.cosh(E/2.0)**-2 ,Mass/Temp,Mass/Temp + Cutn,epsabs=1e-12,epsrel = 1e-12)[0]
# </editor-fold>

# <editor-fold desc="SM thermal and more">
###gstar_energ is from the numerical results from arxiv1609.04979
gstar_enrg_data = np.loadtxt("gstar/gstar_enrg.csv", delimiter=",")
gstar_enrg_raw = interp1d(np.log(gstar_enrg_data[:,0]),np.log(gstar_enrg_data[:,1]),bounds_error=False,fill_value=(np.log(3.363),np.log(106.75)),kind='linear')
gstar_enrg = lambda x:np.exp(gstar_enrg_raw(np.log(x)))
##gp_raw data is generated from integrate eq(A11) in arXiv:2005.01629
gstar_p_data = np.load("gstar/gp_raw.npy")
gstar_p_raw = interp1d(np.log(gstar_p_data[1]),np.log(gstar_p_data[0]),bounds_error=False,fill_value=(np.log(3.363),np.log(106.75)),kind='linear')
gstar_p = lambda x:np.exp(gstar_p_raw(np.log(x)))




def _rho_SM(T): return Enrg(T,0.0,"B",gstar_enrg(T))
def _p_SM(T): return Pres(T,0.0,"B",gstar_p(T))
T_data_raw = np.logspace(np.log10(np.max(gstar_enrg_data[:,0])),np.log10(np.min(gstar_enrg_data[:,0])),num=100000)
rho_data_raw = np.zeros(100000)



for i in range(100000):
    rho_data_raw[i] = _rho_SM(T_data_raw[i])
T_SM_rho_raw = interp1d(rho_data_raw,T_data_raw,bounds_error=False,kind='linear')


# from matplotlib import pyplot as plt
# # aa = np.logspace(26,2,1000000)
# plt.plot(rho_data_raw)
# plt.yscale("log")
# plt.show()


def _T_SM_rho(rho):
    if rho > np.max(rho_data_raw): return (rho/(106.75 * np.pi**2/30.) )**0.25
    elif rho < np.min(rho_data_raw): return (rho/(3.363* np.pi**2/30.) )**0.25
    else: return T_SM_rho_raw(rho)

def _T_NR_rho(rho,Mass,Spin,Degf):
    if Mass != 0: raise ValueError("do not support non-zero mass in this version")
    else:
        if Spin == "B": return (rho/(Degf*np.pi**2/30.))**0.25
        elif Spin == "F": return (rho/(Degf*7/8*np.pi**2/30.))**0.25
        else: raise ValueError("particle type not accessible")

def _Hubble(_rho):
    return FAC * ((_rho)*8*np.pi/(3*Mpl**2))**0.5
# </editor-fold>
