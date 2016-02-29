COMMENT
Since this is an (now conductance based) synapse current, positive values of i depolarize the cell
and is a transmembrane current.
ENDCOMMENT

NEURON {
	POINT_PROCESS ISyn_cond_based
	RANGE del, dur, amp, i, V_R
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
}

PARAMETER {
	del (ms)
	dur (ms)	<0,1e9>
	amp (nA)
	V_R (mV)
}
ASSIGNED {
		i (nA)
		v (mV)
		}

INITIAL {
	i = 0
}

BREAKPOINT {
	at_time(del)
	at_time(del+dur)
	if (t < del + dur && t >= del) {
		i = amp / 10 * (v - V_R + 10)
	}else{
		i = 0
	}
}
