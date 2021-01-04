import numpy as np

PI = Pi = np.pi
Abs = np.abs
Log = np.log
Sqrt = np.sqrt
Exp = np.exp
Cos = np.cos


def SqrtClip(x):
    return np.sqrt(np.clip(x, 0, None))


class I3():
    ''' This module numerically integrates F according to Eq.(xxx) using the Monte-Carlo method.
        phase space: 1->2
        To use it, you need to set n, F, m123, Tqx, and invoke the generate() function to
        generate n samples, then run the go() function to obtain the final result
    '''

    def __init__(self):
        self.Tqx = 10.0
        self.n = 1000
        self.variables = [
            ("xq", 0, 1),
            ("cq", -1, 1), ("c2bar", -1, 1),
            ("phiq", 0, 2 * Pi), ("phi2bar", 0, 2 * Pi)]
        self.var_rand = {}
        self.V = 1.0
        self.F = None
        self.m123 = (0, 0, 0)
        self.Theta = None
        self.f_result = None

    def generate(self):
        V = 1.0
        for entry in self.variables:
            self.var_rand[entry[0]] = np.random.uniform(entry[1], entry[2], self.n)
            V = V * (entry[2] - entry[1])
        self.V = V

    def go(self):
        m1, m2, m3 = self.m123
        F = self.F
        Tqx = self.Tqx

        #         global Theta,s,E,q,Q2,q2,qc,qc2,En,En2 # debug

        xq = self.var_rand["xq"]
        c2 = self.var_rand["c2bar"]
        q = -Tqx * Log(xq)

        Eq = Sqrt(m1 ** 2 + q ** 2)

        EE = Eq ** 2
        qq = q ** 2  # I could use q2, but since there are p2,p3,E2....this would be confusing
        QQ = EE - qq  # QQ is defined as q.q=Eq^2-q^2,
        qc = q * c2  # q*cos(theta_2_bar)
        qcqc = qc ** 2

        Delta = m2 ** 4 + (m3 ** 2 - QQ) ** 2 - 2 * m2 ** 2 * (m3 ** 2 + QQ + 2 * (qq - qcqc))
        Theta = np.heaviside(QQ, (m2 + m3) ** 2) * np.heaviside(Delta, 0)

        # compute E2,p2,E3,p3

        p2 = ((m2 ** 2 - m3 ** 2 + QQ) * qc + Eq * SqrtClip(Delta)) / (2. * (EE - qcqc))
        p2p2 = p2 ** 2
        E2 = Sqrt(m2 ** 2 + p2p2)
        E3 = Eq - E2
        p3 = SqrtClip(E3 ** 2 - m3 ** 2)

        dotq2 = Eq * E2 - qc * p2
        dot12 = dotq2

        dpdx = (Tqx) / (xq)
        E1 = Eq
        F_val = F(E1, E2, dot12)
        # note that p1**2 has been -> q**2
        f_result = qq / ((2 * Pi) ** 3 * 2. * E1) * \
                   p2p2 / ((2 * Pi) ** 3 * 2. * E2) * \
                   (2 * Pi) / (2. * E3) * F_val * dpdx / Abs(p2 / E2 + (p2 - qc) / E3) * Theta

        self.f_result = f_result
        self.Theta = Theta

        f_mean = np.mean(f_result)
        return f_mean * self.V


class I4():
    ''' This module numerically integrates F according to Eq.(xxx) using the Monte-Carlo method.
        phase space: 2->2
        To use it, you need to set n, F, m1234, Tqx, T2x, and invoke the generate() function to
        generate n samples, then run the go() function to obtain the final result
    '''

    def __init__(self):
        self.Tqx = 10.0
        self.T2x = 10.0
        self.n = 1000
        self.variables = [("xq", 0, 1), ("x2", 0, 1),
                          ("cq", -1, 1), ("c2", -1, 1), ("c3", -1, 1),
                          ("phi1", 0, 2 * Pi), ("phi2", 0, 2 * Pi), ("phi3", 0, 2 * Pi)]
        self.var_rand = {}
        self.V = 1.0
        self.F = None
        self.m1234 = (0, 0, 0, 0)

    def generate(self):
        V = 1.0
        for entry in self.variables:
            self.var_rand[entry[0]] = np.random.uniform(entry[1], entry[2], self.n)
            V = V * (entry[2] - entry[1])
        self.V = V

    def go(self):
        m1, m2, m3, m4 = self.m1234
        F = self.F
        Tqx = self.Tqx
        T2x = self.T2x

        xq = self.var_rand["xq"]
        x2 = self.var_rand["x2"]
        c2 = self.var_rand["c2"]  # now c2 is defined as the angle btw. q and p2
        c3 = self.var_rand["c3"]
        q = -Tqx * Log(xq)
        p2 = -T2x * Log(x2)
        p1 = Sqrt(q ** 2 - 2 * c2 * q * p2 + p2 ** 2)  # derived from p1_vec=q_vec-p2_vec
        E1 = Sqrt(m1 ** 2 + p1 ** 2)
        E2 = Sqrt(m2 ** 2 + p2 ** 2)
        Eq = E1 + E2
        EE = Eq ** 2
        qq = q ** 2  # I could use q2, but since there are p2,p3,E2....this would be confusing
        QQ = EE - qq  # QQ is defined as q.q=Eq^2-q^2,
        Theta = np.heaviside(QQ, (m3 + m4) ** 2)

        qc = q * c3  # q*cos(theta_2_bar)
        qcqc = qc ** 2

        Delta = m3 ** 4 + (m4 ** 2 - QQ) ** 2 - 2 * m3 ** 2 * (m4 ** 2 + QQ + 2 * (qq - qcqc))
        Theta = np.heaviside(QQ, (m3 + m4) ** 2) * np.heaviside(Delta, 0)

        # from q, compute E3,p3,E4,p4
        p3 = ((m3 ** 2 - m4 ** 2 + QQ) * qc + Eq * SqrtClip(Delta)) / (2. * (EE - qcqc))
        E3 = Sqrt(m3 ** 2 + p3 ** 2)
        E4 = Eq - E3
        p4 = SqrtClip(E4 ** 2 - m4 ** 4)

        dotq2 = Eq * E2 - q * p2 * c2
        dotq3 = Eq * E3 - q * p3 * c3

        s2 = Sqrt(1 - c2 ** 2)
        s3 = Sqrt(1 - c3 ** 2)
        phi2 = self.var_rand["phi2"]
        phi3 = self.var_rand["phi3"]
        dot23 = E2 * E3 - p2 * p3 * (s2 * s3 * Cos(phi2 - phi3) + c2 * c3)

        dot13 = dotq3 - dot23  # p1.p3=(q-p2).p3
        dot12 = dotq2 - m2 ** 2  # p1.p2=(q-p2).p2

        dpdx = (Tqx * T2x) / (xq * x2)
        F_val = F(E1, E2, E3, dot12, dot13, dot23)
        # note that p1**2 has been -> q**2 according to 2020.08.15.2
        f_result = q ** 2 / ((2 * Pi) ** 3 * 2. * E1) * \
                   p2 ** 2 / ((2 * Pi) ** 3 * 2. * E2) * \
                   p3 ** 2 / ((2 * Pi) ** 3 * 2. * E3) * \
                   (2 * Pi) / (2. * E4) * F_val * Theta * dpdx / Abs(p3 / E3 + (p3 - q * c3) / E4)
        f_mean = np.mean(f_result)
        return f_mean * self.V

