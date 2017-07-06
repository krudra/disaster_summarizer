CLASSIFICATION
----------------
Generate_Features.py generates feature vector for each tweet present in the input file
How to run: python Generate_Features.py <input> <output>
input is a text file with one tweet per line
output is a file with feture vectorts written per line

SUMMARIZATION
---------------
1. Extract content words from tweet using content_word_extraction.py
2. Run NCOWTS.py on the extracted file

set Tagger Path, install gurobi solver, collect place information of respective location (where the disaster occurs) before running this code

1. python content_word_extraction.py <ifname> <placefile> <output_1>
2. python NCOWTS.py <output_1> <breakpoint> <placefile> <keyword>

ifname => original input file (one sample file (sample_situational.txt) is provided to check format)
placefile => contains location information about the place where disaster occurs
output_1 => output from first stage
breakpoint => contains information about tweet ids, line number at which system should produce summaries. One sample file (sample_breakpoint.txt) is provided
keyword =a It is a keyword used to provide names to output files
