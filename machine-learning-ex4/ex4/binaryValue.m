function b = binaryValue(val)
	% BINARYVALUE - return the output based on the input probability 
	% In a sigmoid representation, if x < 0 implies y < 0.5
	% We consider y > 0.5 for a sigmoid when x >= 0
	% The value of y  is the probability that the output is the binary value 1
	if(val >= 0)
		b = 1;
	else
		b = 0;
	end
end