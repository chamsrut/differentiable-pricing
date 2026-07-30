if(NOT DEFINED PRICER)
    message(FATAL_ERROR "PRICER must name the dp_american_pricer executable")
endif()

function(expect_exit expected_exit)
    execute_process(
        COMMAND "${PRICER}" ${ARGN}
        RESULT_VARIABLE actual_exit
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
    )
    if(NOT actual_exit EQUAL expected_exit)
        message(
            FATAL_ERROR
            "expected exit ${expected_exit}, got ${actual_exit}\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}"
        )
    endif()
    set(CLI_STDOUT "${stdout}" PARENT_SCOPE)
    set(CLI_STDERR "${stderr}" PARENT_SCOPE)
endfunction()

expect_exit(0 american put +100 100 1 0.05 0 0.20 +128)
string(JSON output_type ERROR_VARIABLE json_error TYPE "${CLI_STDOUT}")
if(NOT json_error STREQUAL "NOTFOUND" OR NOT output_type STREQUAL "OBJECT")
    message(FATAL_ERROR "successful CLI output is not a JSON object: ${json_error}")
endif()
foreach(required_key price steps adjacent_step_average adjacent_step_gap)
    string(JSON ignored ERROR_VARIABLE json_error GET "${CLI_STDOUT}" "${required_key}")
    if(NOT json_error STREQUAL "NOTFOUND")
        message(FATAL_ERROR "successful CLI output lacks ${required_key}: ${json_error}")
    endif()
endforeach()
string(JSON parsed_steps GET "${CLI_STDOUT}" steps)
if(NOT parsed_steps EQUAL 128)
    message(FATAL_ERROR "successful CLI output reported steps=${parsed_steps}, expected 128")
endif()
string(JSON parsed_price GET "${CLI_STDOUT}" price)
if(NOT parsed_price GREATER 5.0 OR NOT parsed_price LESS 7.0)
    message(FATAL_ERROR "successful CLI output reported an implausible price=${parsed_price}")
endif()

expect_exit(1 american put 100junk 100 1 0.05 0 0.20 128)
if(NOT CLI_STDERR MATCHES "spot must be a base-10 number")
    message(FATAL_ERROR "malformed spot did not produce the expected error")
endif()

expect_exit(1 american put 100 100 1 0.05 0 0.20 -5)
if(NOT CLI_STDERR MATCHES "steps must be a positive base-10 integer")
    message(FATAL_ERROR "negative steps did not produce the expected error")
endif()

expect_exit(1 american put 100 100 1 0.05 0 0.20 18446744073709551616)
if(NOT CLI_STDERR MATCHES "steps must be a positive base-10 integer")
    message(FATAL_ERROR "overflowing steps did not produce the expected error")
endif()

expect_exit(1 american put 100 100 1 0.05 0 0.20 16384)
if(NOT CLI_STDERR MATCHES "adjacent-step CRR estimate requires steps below")
    message(FATAL_ERROR "CLI ceiling did not produce the expected error")
endif()

expect_exit(1 bermudan put 100 100 1 0.05 0 0.20 128)
if(NOT CLI_STDERR MATCHES "exercise style must be either")
    message(FATAL_ERROR "unknown exercise style did not produce the expected error")
endif()

expect_exit(2)
