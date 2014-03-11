COMMENT
Since this is an electrode current, positive values of i depolarize the cell
and in the presence of the extracellular mechanism there will be a change
in vext since i is not a transmembrane current but a current injected
directly to the inside of the cell.
ENDCOMMENT

NEURON {
        POINT_PROCESS ZAPClamp
        RANGE del, dur, pkamp, freq, freq_start, freq_end
        ELECTRODE_CURRENT i
}	

UNITS {
        (nA) = (nanoamp)
      }

PARAMETER {
        del = 0		(ms)
        dur = 20000 	(ms)
	freq_start = 0  (1/s)
	freq_end = 15	(1/s)
	freq 	   	(1/s)
        pkamp=0.15 	(nA)
}

ASSIGNED {
        i (nA)
}

BREAKPOINT {
	   at_time(del)
       	   at_time(del + dur)
       	   if (t < del) { i=0 }
	   else{ 
              if (t < del+dur) {
	      	 freq = freq_start + (freq_end - freq_start)*t/dur
              	 i = pkamp*sin(2*3.141592*freq*(t-del)*(0.001))
      	      }
	      else{ i = 0 }
	   }
}

