(
    for /l %%x in (2, 1, 2) do (

        for /l %%y in (1, 1, 11) do (

            pypy3 run_search.py -p %%x -s %%y

        )
    )
) > output_p2.txt