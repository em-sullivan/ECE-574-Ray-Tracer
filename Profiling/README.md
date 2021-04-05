https://github.com/jrfonseca/gprof2dot for more information on how to use plotting script. Need perf installed first.

# Example commands to run : 
`$ perf record -g -- /path/to/your/executable program_args`
`$ perf script | c++filt | gprof2dot.py -f perf | dot -Tpng -o output.png`

To install perf when using Windows Subsystem for Linux (WSL), follow the HOWTO at https://mirrors.edge.kernel.org/pub/linux/kernel/tools/perf/, using the most recent version.
Doing it this way, the folder perf-version-number/tools/perf should be added t your PATH.

# Callgrind and kcachegrind 
`$ sudo install valgrind kcachegrind`
`$ valgrind --tool=callgrind path/to/your/compiled/program program_arguments`
`$ kcachegrind calgrind.out.12345` 

If using windows and using WSL, you will need to install an X server to use kcachegrind. I used https://sourceforge.net/projects/vcxsrv/. 
You will need to add export `DISPLAY="$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0"` to your .bashrc or .zshrc.

Note: using Callgrind takes A LONG TIME! It will give you more info than just the perf graph, at the expense of time.
