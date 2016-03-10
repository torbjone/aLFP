COMMENT
Since this is an (now conductance based) synapse current, positive values of i depolarize the cell
and is a transmembrane current.
ENDCOMMENT

NEURON {
	POINT_PROCESS ISyn_cond_based_balanced
	RANGE delay, dur, g_in, g_ex, i, V_in, V_ex
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
    (uS) = (microsiemens)
}

PARAMETER {
	delay (ms)
	dur (ms)	<0,1e9>
	V_R (mV)
	V_in (mV)
	V_ex (mV)
	g_in (uS)
	g_ex (uS)
}
ASSIGNED {
		i (nA)
		v (mV)
		}

INITIAL {
	i = 0
}

BREAKPOINT {
	at_time(delay)
	at_time(delay+dur)
	if (t < delay + dur && t >= delay) {
		i = g_in * (v - V_in) + g_ex * (v - V_ex)
	}else{
		i = 0
	}
}
