TITLE Test persistent Na channel by TB Ness

NEURON {
	SUFFIX Nap
	USEION na READ ena WRITE ina
        RANGE gNa_pbar, gNa_p, ina
        GLOBAL pinf,taup
}

UNITS {
	(S) = (siemens)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gNa_pbar=.00004 	(S/cm2)
}



STATE {
        p
}

ASSIGNED {
	v		(mV)
	ena		(mV)
	ina		(mA/cm2)	
	gNa_p
        pinf      
        taup
}

INITIAL {
	rate(v)
	p=pinf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gNa_p = gNa_pbar*p
	ina = -gNa_p*(v-ena)
}

DERIVATIVE states {     : exact when v held constant; integrates over dt step
        rate(v)
        p' =  (pinf - p)/taup
}

PROCEDURE rate(v (mV)) { :callable from hoc
        pinf = 1/(1 + exp(-(v+48)/10))
	if(v < -40){
    	     taup = 0.025 + 0.140*exp((v + 40)/10)
        }
	if(v >= -40){
             taup = 0.020 + 0.145*exp(-(v + 40)/10)
	}	
}