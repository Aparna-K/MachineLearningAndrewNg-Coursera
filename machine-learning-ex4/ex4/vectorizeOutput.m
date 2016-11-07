function vectorRep = vectorizeOutput(val, k)
	% VECTORIZEOUTPUT - Given a val(label) of the output, this function
	% recodes the label as vectors containing only values 0 or 1
	% We are assuming the val is a value from 1 to k
	% where k  is the total number of possible labels
	% i.e k = 10 where we are trying to detect values between 1 to 10
	vectorRep = zeros(1, k);
	vectorRep(val) = 1;
end