TITLE passive membrane channel

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

NEURON {
	SUFFIX passive_vss
	NONSPECIFIC_CURRENT i
	RANGE gl, vss
}

PARAMETER {
	gl	(S/cm2)	<0,1e9>
	vss		(mV)
}

ASSIGNED {v (mV)  i (mA/cm2)}

BREAKPOINT {
	i = gl*(v - vss)
}
