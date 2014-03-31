TITLE Borg-Graham type IM channel
UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
    e_rev = -80 (mV)
	celsius (degC)
	gkbar = .01 (mho/cm2)
    vhalfn = -43.   (mV)
    z = 3.5    (1)
    gamma = 0.5   (1)
    tau0 = 1. (ms)
    K = 0.004 (1/ms)
    R = 8315
    F = 9.648e4
}

NEURON {
	SUFFIX Im_BK_frozen
    USEION k WRITE ik
    RANGE gkbar
}

INITIAL {
    LOCAL a, q10, b
    :q10=3^((celsius-30)/10)
    if(v == vhalfn){
            v = v + 0.0001
    }
    a = alpn(v)
    b = betn(v)
    ninf = a /(a + b)
    taun = 1 / (a + b) + tau0
    n=ninf
}

ASSIGNED {
	ik (mA/cm2)
    gk
    ninf
    taun
    n
}

BREAKPOINT {
	gk = gkbar*n
	ik = gk * (v - e_rev)
}

FUNCTION alpn(v(mV)) {
  alpn = K * exp(z * gamma * (v-vhalfn) * F / (R * (273.16+celsius)))
}

FUNCTION betn(v(mV)) {
  betn = K * exp(-z * (1 - gamma) * (v-vhalfn) * F / (R * (273.16+celsius)))
}

