#!/usr/bin/env python3
"""optimize_cleanup_stage5.py - staged, abortable optimisation of the Cleanup project.

STAGE 5 (run AFTER stages 1-4): compositing, fill_repair, scoring and selection of the images of a batch
run in a thread pool (POSTPROCESS_WORKERS = 6, CLI --postprocess-workers; 1 = old sequential behaviour;
result order and values unchanged).

Stage 1 applied only changes that are verified to leave every score,
winner and mask unchanged:
  * pixel-identical candidate masks go through LaMa / compositing / scoring once
  * the mask set is built once per image (was twice)
  * FILL_REPAIR switch + fill-repair metrics in the reports
  * per-phase timers and LaMa job counters (run_stats.json)
  * --no-debug-images production switch

The restructuring into src/cleanup/ is deliberately NOT part of this stage: it should
follow after the baseline measurements, see the final report.

Usage (works from any directory, stdlib only):
    python optimize_cleanup.py                 # inventory + dry run, changes nothing
    python optimize_cleanup.py --apply         # backup, patch, validate
    python optimize_cleanup.py --restore BACKUP_DIR

Never touched: originals, test_runs/, diagnostics/, .venv, yolo_mask_winners/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

EMBEDDED_PAYLOAD = {
    'yolo_mask_auto_select_modules/pipeline.py':
        'eNrtfdty40aS6Lu+ogaOsw3YIFrdHk946KEj1C3arRhZ6pVkez0aBQIiixIsEOACoC7m8m3/Y//iPJ03/9jmpQoo3Knu9p6XnYhx'
        'iwVUVlZWVt4qK2FZ1jSej/JkJOO5+OX0+FR8IY6DHwIxC+J5OA9yKW5kLNMgD5PYFfI+iNb0t4Dn4iGM4ZlYyTQLs1zGMwlN+a24'
        'DvLZbRjfeJZl7S3SZCl8f7HO16n0fREuV0maQ/84yQlUtren2mbZvf7z1yyJ9d/LIL/VfyeZ/iu7XedhVPxaX6/SZCaz8vlT8Wcu'
        'l6tFGMni920qgzkgWDQ8rWTGmM6SeLZOUxnnHqOcaYwvqNf7JImmj3K2zpO06B4upe4dRXJGs9Ld5nIRrKN8Hs5yfmcF04nCa/38'
        'Pc6OHgAWgJNuP4ifSsLcv9Z/xuvl6kkEmYhX3Ov90bHucrQMbuQeN3vFCvrLILsr0LH3BPyP3vwB2s9l7orrdRjN/RDb6GU/w1YZ'
        'L5J0Jn2ga85zcsVNmqxXfjgH8oSzIGLQrshy7JondzJ29xyNQBTqQVdBmkk/SG8y/SyJF2Ex1c9V61xer2/8ZJ2v1rl+xrjxE8It'
        'uZdpFDzBoMG9VA8AXB7Mcj+7lYi58SQDXP3rJ/6XHquxbmSylHn6pMfBtdVTlXP/tySWMDFY/iS6l/UHPnIqwwkTH/mwRt/ZbZLA'
        'jHkq/jxMXbM5haVJln4WLFeR5CfzMJvhzPwsWQPRM25dhDGsS7IKwjjnFl6kOFhKH8fnxmVwB5NMHv0kjp6IStxOcwpiGDVFjI1G'
        'fLn2HqxsDuwH/z5kunkJKPnZMogiP4hWt4EfZhHgrp/fXMOaprAW9PMhDYHZYJ/lMpV63Kzkh1QigQpC8euAlo8yw1/HwGDQW0Zz'
        'zSQAKvKhVxCm5SrhL38RBblPj4M0Dxew9LoPUDE1NhGvRhgDS+ewp29goGWwcpliyDVpkOVMCW6Dyaa5jx24EYgShbH0IxlAEzBM'
        'OFOzx5EAht5lwHUSt77RAjPfm0VBlomzdXwOwi4bU0+Qij8DSUezKJndofQUq3AlcRgBNM7gv9E6E78m1yBO1jFQMxM2UgskrMgT'
        'ka5jmA1A81BIOiRkmYfkAgRtGIe579uAzcIRo2/FCTAsj0tYQ7Pn08CTUg56x9BgO9W3MgkEmmdjgaLrEujnikWUBPkV9DSEmk2N'
        'tb6EeKUrcHC9IzTVupHiAOxuxiICjXJZdgdpeIX9L6/KuQbzOU3TZbKNBb1ZoM14NUlAKqqkQ/mgPvNLAnslvphooOXYKPLTxuiO'
        'QWhcITEhgHtFK7ODf4G9qwPz4klcbrV6PmlXp/qaxpEfevk+LiNA84CNFr7iF1jJNtiPBWNwb1d8Lh9nbQMQdyF5aWpu2whiVMXD'
        'GDKVoDdjNUvEZQ8RyGA/giSE7XYnn2wwJMa4qLQ+wAiMBBEsT/Epc0YI/bwwm4c3YW4bmKohoKOdlW+KyURY5z8cnF34Ryd/txqv'
        '//Wvf/16r/r7r4zcEiSsXcUFRP6vsJ8BI9TQto/yBuwXx1M6wXY8UGtoJ/A/1AlVHPQo9Z2twDCSSib6aALAa/iCp9o8+bgCybHO'
        'kGTlGKXQz1CN6E5lS1+/Bxne3ObV4VRbX7dgPQdOyWZpuMLpa0K8FNZTEiWshbW2oHe91ZMSQbAKNArp9zlZhH+aiH1v/6tyMVYp'
        'Lps1PTs7PRuDDAozcY8WJBiVgBSMlInRyACAvVE6s00JHSRYezM09RbrCGy7LPcsEE2wOhOw+bwsn8s0depr/7qCX7i8yX4TfxNf'
        'vt4XScpt8zBi0/ZvomyMwmWYV1oqyhufdM1sNOJRlusMzBgpvp3AaN9Asx7Ihb8JvkvW9GjEoEcKdNlvf/cJgkVT4oO7DdhL0raC'
        '7Sfn9gYxfAQYsLi24wgw8cQjML0mgergZasIdpzlWg7SrOiwLYeF5hI+LPLlviteu+LPrviLK752xav9q6pkSYMQdNtP4EHIaZqC'
        '9YyNIIAkcFnZ2k3NcjRNGfkImj96Evvua/fP7l/cr91X+zQjk6l2ox2bZ9ogQMWY3SZgiqB4XcNqsJZDwbSWTDX6Eyn39vTk4uzg'
        '/ML/4eD87/7Fu7Pp+bvT48NzpwqztMdDAntZILGwChAXtD6wueI5jyU+B0LuO854//V8a1UHbkGXYF5pVreLIYL4yQZ/C/eS+NtE'
        'gYA/Xnn7gzBLasGLkYxBg+R2+7SAoYAT8J2O5wTKqa9xhU30gneRtVh9sBj/HZAOFjn7oDMQwiAVBVEPJYVVhYt7TAazWzVZDeZa'
        '5g8SLKt92oSvgF0q3WqsUz7sYqLKDiR7Fq31zFXCn+3rQhOAa0KyHBiiZq3bprIoBwvuAxAf1xE7AWxh5HYdXGWbNsVWdWOWQhF6'
        'Nq2B+r61rR/htwQrCuiOQg7RT9K6/HIFMtx1kt9aTutwJPqePRwL5GcMl6Rz8kc0udp8MLuBRY3MbgsRVdsyjP0lU12/lkk5r0As'
        'MZJRJhsWJ3mWeiWrAzvPWikJGyJjWGS0lC87HoJr9ACAZSckIIgAjVATfuvyLKyTBLdUOFdMLZDDSaowa2yKEbYIfmPy9ba2WDWC'
        'GF1711QpN93TFWBiTkBleTMwXxcgxJ7FgXXold+X47Lr1V7HIlZ6VPYi0rjytF9H2khc5q1iDM+gGZBmScqElhsBOh62EfnxJ1K8'
        'Mt6ViQ1JcwRRyG3EnlucIczO9ZzVsis0KRYDo85BbYYzjCAykuR2zcPFQqK1DLob3FnUz9k3bHAKDlM9JOtoTnE0MP8qU6XACHEN'
        'zrgeMLFLk9gVJSMU5ncBh2zYVSpnt3J2ZwhR7mFSp+XNhuDE9TTtZXBWyFGwnS7J9h08PUny71BL6R10BmwewhIJZVYLZXwvwyyD'
        'n2PYNMYY26ZARTRMa//D0KAAMM12lYBeJqgLfAEQMKEjAgVYDogpFwM97doTrZLMYECbc4/e/WZb3yu02eKkHoCrzqqCQk/IrkXM'
        'G7JICe9avw45rlA0RgaLExGuxhraSaGVfW0ou4TWlIrr+C5OHmJT2OV2FSz65L3qQyGtQLXLd7bIFtbPB2cnY1FgyDMgsbJBOaFg'
        'OFve9uCFa/WhSCrenv80rplg1f8trM0LV7zwfgVe0/Aux19fOds2i71kDA6HAiHM6KgRvCqDrqW2L9tYJ/JvQzwYvsiU/qEzjgzb'
        '6tbqQluoGzw2sDGE4vkckvW30AoN2919Dg55ZeFvyBU3Mg/yPLWx2zKZryPQNxr0lSus44MfDvw3Bxdv3/nnR/+Ywiivv2LAFPKe'
        'RTKISQkNATqcvvnxe//84Kep//Z4enAyPfSPfjj4fnoOIC9SdHHQGDbfOjg5PDo8uJieG8Mly1WQhlkSP3PE0x/eH5wdnZ+e+Ofv'
        'ptOL54ypov/Z80Y8O/iZ3YjTn6Znxwe/4DS/C8AU22lM8wzheeOeHx1O/Te/8L96rjsNrE0UEH18kMHqzYgvVhe8uSB1crVNhVAh'
        'kOtM+ma8HZ4dHR/7Z9P3B0dn/vTk4M3x9JBwNqWy0aOAMpeg+aH/4fTwx/fHR29hSj5M/+QC/jymVThvwKE+e2bYVEfM1b7GmHdn'
        'rJNDnACBfKvB0LHexYX9okjLgq1qum2F/QY36Oj79Pf/+/t/gRLblBt2K96E0RydzzQRh2CF3EbBeuEo7ayHea9PR8ESOlhnN8E1'
        'QikFktbmO2gTbaSQumAC4MKulzZiXlUGHrCpHbswaxXliRumoNMQbOez23X+GwKKAccSHSDDpmXwrfgHvoqAkQwwu4fgNtIEqLo6'
        'KppTGeFOhrGskQvskmh0Riw12dS4cusKPLOUMa02P6Y/tzUgh7hlRqckxXGgt7xPJhtz1wC0t8WGGZ3j2aB+oWiGd07VFlLP9I6C'
        'J+ewiUZvnkb4r3pq7q/CStJ7eZ4++cDIDZocnv0izn4EbTuXK5xbPAtB1eIeYdtWsM1LhpmcfwOMARQnM3pGjrc6MEWFA2+BZVFz'
        'EpTC2WdsSs7zlneoEDl8nU1QBrug8DBEn9zRT8fswfb5xNSvL4VFrVbjvV1h86Ggtt2qoDm9wOdXPLDfLLXVUR72dVSHwjrylFnY'
        'WAai6OQuY4DNkA35KTpjwLuQeJAYpE+HYJ/P8iR9ssEjWISPEyMYHqzzxFfHf5aDZkO+XOGJI9ruFQsI2rVvXr5SNdIIbLHZOv0T'
        'ggbI6XNWlI0A/CXGudimhU3vI7CScoXVS84Er03ZrxLSb+sAz/EUwei8V7eHW5BvmpntB8W2ORu3O3LWMkTTymUkZ8t5JdBaiTI8'
        '4RkI5nGgtezSsZPpZDluazdLBbyyEfCaxd1Kz7O7k5qX6mFOtbuPcrlUH9MB6+7DG8Ey58NL1Tefm+w33aM4HOl+fxnGdDwDXax9'
        'b//1V1bnq/qkw4Su25yeXirQjyO4FjsIy2Blkzovnjo9EOR9OJOW8uj4V+fLcTK6D7M1mEItE7lqc6B29fqqnPgFsCIMV7hUiiiG'
        '0+f2eEsGwgWE0RLmF2rqvj87vZi+vQBT/h+nJ1OwtM6+Pzrx3/+b05yD1pP/jC9fvfzySrzBJBuMO3AEoMj9YpFfOH1tW085gJnn'
        '0UFLtM5uDfFe2fZ07k5Hy3x6zhK0KnacLq80mZGpo7O8PFCkdkFbV8we5hPlzLmsKSdsZ7etH8LwWCvOkjkfUfZ7w8rj0+EZDlgt'
        'wNWWc54XwdnUAPe6gdXALWnoOl5A91d7LVFe0nylegPCoMlb2e5eehMl13Yh1A0l6rTSBLm4AbkvBNwax1L0YUA6fCXgDTAPmWgs'
        'oIqIFmPbFgY2FDzzBkxzGTzaDSQ52hsFy+t5IECzLMf0Xw9ZzcZwqL9EfutSEqzVGqOZp/oVtOB92YjkmMlTdgNWdejmEUAr3M22'
        'uvQ6ZavIAyiwB92tk7+smknW2nkX26zSm4wcMkJVEI+ORMu8HpcsmloAjw4/y8M/5WORbwYdTYesCAh1xaw7qKWSRdp5tEJOdoYA'
        'lAsYOvTL4jey9Q3gpXSUBcYb4GA1o1ZOcxsitnNyfbRC6hAhbKHanbK9WMjON9B6BYXr7+9/ZfW8tLA0Kv5mzgfGfSBb2KQOb1M7'
        'Vth6q/imvUsLhQzJMhSTHhItvYpRB64raquMnRMHTDY4iz+hC2kNANNEnGzm29WjS7hPNuQGD3fmxSTemmyYS4FPXvxNYfPti541'
        '6SBhufsuFQ/PnStKMspvmz3qW84LVuhU8iI208IKJ4b8BXqLM9kwF4r2OqaeinjlxXPK9bxqO3bM1lE+bn+9KRSes3lIt1KyspfA'
        'NOxWWpDHRU/CZQ9rEZaXc8QI8Asyws9W/TBjCTx7kAvHlgNQMbQ7gdfWMJ+vHfEteM8dSpvhNimrE6mrdHUxa3ZsEAhsVMysNZuI'
        '9maG9rhr7LbE7Q55eHM9gf+r0Sb033ZzE1SFzCb1hAJDgl5edRiqZMlO2sJQ/f1KHpy0sKPTiWaZ7jypJDXbVVXZLcg6QLfkwExa'
        '2gY6VxJfJu3Nbpexq46F2L73V4+TLuPeHVJVn4nJR/+vBvDyNfoM/yIuvyTfAY8jRj9Nzw7O3kyPLn48+V4cnYh3v//n+/dv301P'
        'hH3+9t3pyYV4iWGGH08OxdnBD84fi2Ke5EGk5J/gc+2OwCf+L14vfQrq0st42cSbyTCyK1BeGgc11e4BhyfR4hsKP9e7qSsKlEi+'
        'U+i6xX+rrMV7do7Q5qZbPBhD5DA3Hmdh8oUx2a14w3+Ma45bwyi7Bl6lJME0iG+kbcBoTx7GDHrAmrt9blCu8bLEEzykeoiJCNzx'
        'C6ODW1nLpgnPb7Ynaih4YxqjzwcejUZMCrFhlL8Qr7Yvq6SyVah/U2L5ajvaEOytuAfqbkxMtyDDR6M+uhp5vXR7pozDgScT+GEM'
        'XhIHLvmtptnAEm6gc4eRlwYPrT3ZO2vpYKK6a2y3A9kP6V7g+2zfRbGkujTT7sCI/0DVDG6M0YwNHcaLYk9ZveBgKuzejnSHqtJV'
        'G06Mh4nSVYsj2B5LIfVf3tXzYesTxULUc+1WkelvGRupx4QqDAj0e/XtH9vMzLlEEFdOJwhKTpqAi8VRPdhz4CHM0DcIIthwNjlg'
        '2BoLa+Rb5CwLy+fMV26vqfLusbRFRFK9Zoy5xmS6IZQrzfMyIHX24eVFj6/1spytAZQX9DLKsS4Os3YIAhJVNrNwPhbwH6ZNSPZ0'
        'C/Tt0PwY42KG/HOvs9NnYgqW8hPo8qN//XFad7vsespIEKFx9oSxO7CA8IEjbhJ4kCekpHrG0fndYN6HOYZaowBzffF4LslCTj2b'
        '80Ws+TdlwlkG5jFQk24KsHXudQ6iSOeKGWHfTkEP40lZn99a3prhm0+X5XFT5pNisOge0ateCMAFai0ApSsMS+ISD/JD58DETkyT'
        '+S7jVw/Stee42YEhhVJT1ljw7irRsMjvsWEm4FdZ5W7IE/XEmPJOEXDBWs1fhY+wDwAI6nHwnogAfpzEv8k0sWk9nR0A9mwP06YP'
        '47V8xvKTNiU0f02us12IT7dnKaRV3KUlog0NegMGD8gyzIeyNuTb+P6Gum+xvTtiU4Oh5XoTCj0ZhsNuOl7DZLVAAnaZzOXEOvv+'
        'jQXSOriXdsXmeWlMwHkWcHZgFXhw2Uv/fXCsAZlfaA4diK0PTZzlgS+H2V9liOBz8fqrrwyMdhyAMTVtud2JshOUHaZr2Ec6rIKc'
        'h8rALnFxaxAbIInn1ZmrvrQsrHSNodUFHhwCaaLJhrPm8e+tesAHdOoJQeGWreUOhNvUaexkYy61BkuRgY1BFP2ArdzJprAqt1bT'
        'P8h9xKQn5ajiQ/oPSXon00wdUrwiI84uJ6SOrfDUcgdjjq14JmGX9aYWg7M+iqzQlU+J/qTXVj7qs4rlDoosRfvZaWeIyjTAPCt+'
        'uypVvByyAwLGewsgeMGoW321n+lpJnrOkV5BSBrXx7MzSoqvnadd7XXZUuMeI+QMo9CjNLnGqxG3QYpXiyUZQDwc5gOifZIJTM8l'
        'ar8kWwLvHeCxF5goYIn0jJDhgukkHu5kI40F+7Xg6qi7TqBAs0SZNLp9dos++bzbwiGcQaEnec3X48lYPbH4hw932Kqjw3pmzRBI'
        '3Q67M0IMmocGbC7bmB3G+bLN3dZ0mp2Pwb13AHasPxp+SR9tbv3PiM5+aVqd9Auc9Av46wVT9cUHAmd53AWbCPqiT0Q7vdwTQkft'
        'zUpwJdERllWR1c9LWBEhFP+nlF8DC4dlPzyskGEbSUBa+S44dE0G7gCPFm87zxqwqux3GY9ZdofhSiVwaUjn93QCUxfNnFyKWl/L'
        'R+Lm7p1eSbbbRYo/eA8B3v6ngR5wGIVfjzQJ4yCKBsapgxt2tEC5PXirJIoAGfBGu5N+GlPyQO+CQkWO7JOVtMLpMk+lNFYRbImb'
        'GGPEEs9Csx6polQqlTPyzkO8pXiCOmQVzKRdKsJJLB9z2575vHZ+SQRFdYzJ+JQWAyy177TZOxxjVNF1w4AkQ7ijokRR3qNwLS1q'
        'Ah+uCLy61RjmuBgKHqiFokZtmOwsiCxVw8Ma8w1hu72oBZt9rvjSaXEMdcwY/JHT9yikR4248VhsNMZbMQ1jzPeOXWxUGG8xSv8b'
        'bCgb9yvG5PsQGXuvFttsILPqY1Ka1ER0MhPg05jS/3iGE9UrUWKH/RuO97VPyLKstzoi5HI0iONCdPAhYJOyVfaNQiIDpV4cubjC'
        'OLsBKze58bEQj+O1x78w1VvIxYKqVKABRvdAgW55wkUHaKQXbAuaedlkuekbkkGMFxqMUmPRk9c1tfakwmceHnXB+IBzq/KSbHiD'
        'UpY9w+LXh0eHjWhtGXYFabtSqQu9sVYzkDnQR+EzlzlwPKr8WnEo25yZ03n2TCZ+GQ72dFv7oXCtFtukLco40DOc6zUqjwd05l8N'
        'fhfW8ye/HXXjQfcJfSrVnYMGgJbH7aTn4ly+jO9lBNaECaP+bK8TgpwrOvBpDRXmMM5M3EpyRb12VR9L64sF5F6NG6cyg0ktheXh'
        'FxHqFjCMtYmjAbgHMhdgA7C5lozXT2U2QXOcxsw7jpI41oEY7xjrMMLmVDGkwqF7nzC2Xacl95w02w3AV70Q+2mo4Q+8tfNog2Hj'
        'zOdgW0D/FrHfZiCu+zCmniBVHuy+VOApO2rVmxrFZkERRzayo1aV1CgKrTaSoz4It+DZuGnN0o9dfx5XNx1nWqjVRCktc7efoShm'
        'qgzAYPXUY+VTn0sej9L4FOF1U98Oqt2HG/CjGmFFo6vlDHsuC9btixQLHwacJN1RfdGumgOq56yorKiEfHE22E2e/nAcoVXgszFn'
        '5M/AQsZiiT6VLAAzf9/bB6eh9RX+LfVbuzoRFWB8/oS/InCvFKhtz7prDq4aS8P8Qq/VOabSuDe0hL46LwnKpeyq7TmQa1sDWJ2L'
        'W+auTZqL3pP42imwCV9vvcIdaauV3/X1ZgHbD5xbi4nbxtTd83P2nqvj7CoyPVPeTaURYfa6zy2x2KNVoEHZIwZMq6sEJBsOTpdV'
        'wc7Y7maFKrCKU4BttbNN82kslJkWWsVJaNvafGIN8gFm0WfiHK0FHu1bPj0okCxaFCk98RbEC6VCIPHEtZwF60wODFAvZQtrTZc2'
        'MbMim63lyzyUo+tUBnd8TzND8xv8255EC2N5qXrPLLerq12xq5xd4FxyHkCY+UWag4Uc8crb38GuhMUOqCCtYfTvZtsNqyisFsy3'
        'khvFg+2K/9kUKxVnTKtQZ0ey1moS24N6rV/EVXfEsJbU+r7uy7minFB1foMgW1zL4U4VumMBJ9gda9RJ2NDffSdKawWzg+z9KEbe'
        'H2Dk2gZSst7cW87uGwGM5DnogtkdBobNdXecRrJDt0dD/rkO79rF/KjdunJVrpW2CxWS3fCqu7Pi/yHafUlqqubD6P3J9yJbyXB2'
        'K9O4Tw6btSGGMr0oibGWiNgvETCjp/caWBmn1iXILYrDgw6rljUehMA5k+WVQAPKm9N/809Pjn/ZEYhO3PKi5IE0PoAKszDmOzs2'
        'LSZeFtoNmnk/DbUc3lBbPVofsR9n0XMKQPDS+saxW5nqROuzHU5yUiPq0s8fefpbTy0qOdvInDrQ6UxqbKeP63W1EhGv05LtwV4C'
        'Ox+rWYF4+S3MxY18+P3/xdnsNh/cEbrgySffEsk9Bm47Pu9Qcyi1xBi4lqSuIBWmOf3sX4LkfzfmH7Axk/tnbEyMC5lrX9+btERb'
        '/XR4j6rB/6A9mty3ZDWqITu2puFiadMIHSyzsFePh0W609nbLcSjauZk+tJPX5znQX0F4FrivTb5qIYCTPDvPIlkijxRVJ8xDEvW'
        '8x3nDAhOQaK/Tf9LNSlbkP4srFfGp52En4nvJT8fUQkokSULLDFAEnNAvyeqKFHJjXwK88wbznVGsIuZdCSDOmZ+KrOJGtvpmmRZ'
        'kAr+pG/acGGqgRkW1kvRfbz3iSS1KYdrZmKf8/Aco5HNOz/UtWyvhrL1n+HUc17qtYx2E/EkzEcozD9KxIM0H5E0/wgRD908FVu0'
        'LR+rE4ysj5bzKN4/UrRrCbNeLMJHvMojLn8+OjmZnl2Zs21sfEWZ/pFNTiicCMCaVnArlAW3qaCAKQhNN6HHp+j6blQ/e1T2gVvB'
        'tN+j3MUyNdGoaj9ttJY72/t1dfPsGgKfiXMKxf1DhpGMBSiZIMtkuzAxfViMr9E1I64o7WfKg/OXINso44sUwacKwaFgarjQPaHP'
        'ZhbEbpdW6pdVCvMHtoZxXaUXxuefb+7k01ggf9IXBcbeXxZb2gNLrnaxwE8xSf3NCMX/nMYGPd3yowtLfcFo2z3ktmNl1V4IM/50'
        'lWjbejvIsBIMmKedQPpFWAnj9uk6xS4tYDDhyXr3y5uzo0P/8DWKNf3jL9Z2AKy+RN8ryw3x2GJ99ApLzBpqxbj99v7e7sKTi81E'
        '1/1qqFwkqpE7vTi6OMIiudNjuvpPvFVb771BRUQLtgs8WPgBaGqldoDFDDAArvqFjx2g6mUYgIvVYADcyXf7+1/5Bz9enJbwnrFk'
        'qhhgUR+or9pQucSfrOJQDy9RCN4HQReFVPv3OkmibrxMI5vxKEpnVCEBOvvevvN8TFIZgF6i1FHKp8p7iQTCmb+l0ytdh7E2Ru3D'
        'vOf8rUOiVuGT6YFmabdGeTUuN5rqPSYfSgTz+4A/SvMFl9ZTkWk6IOhJon+9A8RvKT78er8HzJdtYEA6YhIj/Azna/CdCog9gP7c'
        'BghMkusQq08zYvFcgx6CuO0sFVVlyJ6L6FnMlf4bC0W8MmRj1thn6JJZY+a6Fj/DUJmxTbh91yScjrtIQATazk3+1wvmZ6sEvPNQ'
        '0vlbbfP2ePsBvA7YzdczJc0+aBd2YVEBPrgfe5zTFEDgh4K59M0nRrMO/sMRZQ7sR6vAQizDWZqM6JA2xpq7xfHpeLigGLwmiLKw'
        '066fxKZK6rG3vwCH7ptBOMXUgV3BgvpiU6fF2Pty0Vuc7PlHoUwkC5dH5A/JiJO/+Cxob5ek5GFrvmrJd79GupdT/9kQG3i3qEhq'
        '6utxwyjogVLUSW06Fw0b09kFDtNtjL5hGWJjp8M1XqPg3b8D14X5U6OPPpB70XzrxZWCtQsqBe90DlB7pYDO7NCPZRmLHESpAqcI'
        'WTKsahizGL9yOtyJf/OtHQhUiB3WhHTOTfVY2kbofHmXleAkArCG44wj293jdL27wzBdluK438r7ZGYnCZ5n4GfahONnafoGKD6K'
        'IxYq7ZmxGKpEuWF11Zj9i13gv9BKnNblk1BE8RZnm/wxM+kY4hNMpje7EzmwyWQ9XRQ+zx3PSBV9xpBFr+eN2swpHR6z3md4xLLA'
        'YKWGSnOYxovPh41ckid99OvuMTwa723c61mHLu6Km3HIkFwXjGH/h1AVqVSMfIxPJptll6YEW8llxqeXemU5vpvdBit+tUUxouGl'
        '4qt8bKZirJwkU4ZZe6/WtUVD3bpBhZerhaBauHhTsSTCFmuOchxfbBr2ybbFXMv91a43RlarZnkMLC9RlMgwU6j0xX8uOWHUJXPa'
        'C5ebwDvrTXCJVF05YSIu69cLffWRHL9WDK0Zfe42eWf3r6k8aEDFDuAX8vfJennBbR05r/heVnnvlUMRe9DVsHrgEmZL9LXx62tp'
        'sqRUTj4z/kYE9wl+TBhPxrP1NX+4AqzS9mzcvnvXdJDM478H93NKH8ZI8JL5o6bupCQ0X9uA9/rP3epEp6tq2M3Da3bVFXBFZaVx'
        '/ndSrjCxFO9QUmXH7gtXfVe9m/Q11snZ6zmrr3AlbqRZlMzuek7qVyuntdIx7jzemy7/zTuyvG5a1Gkpbp/tfO3Tow9pzu1yDGcn'
        'z8rsxu1Oz3Xh4l5sUw58hl/6y8D/mt2KX+VcLtUF4uAuR6GZhZi6Key3aZDdjvjzT82RZklUcAjmL9Ix+518ymz4j6YhfV2wOg19'
        'mEKVQ5IHj7o4XTV2jML8fCPJeoDVlHgVGfy9ibXOF6OvR1l4A62xfMAJTyz+oM+ioz41ZkXTdwaze+8QEP+ZGuyF+QGbCc7O6env'
        '0T+3wJCdmfHmi/SRAVXvpyHfjW8hpOxtpqxm2VGmkLNTfBeXaNUmVVu/TEdBvta7x733KUuif1xGDn83CCjt42r66xjrXHL+uT2Y'
        'QVog0W1NGCRtKtIqXakCR3qpiHplEtQlOOT1+9AJ3zIjAFdOl0HTurc+YVVkBngx/eH96dnv/3k2PRHn76dHb9+Bxj88mp7D8lIp'
        'Z/j3/PS707MLcTydwlt/+uOxqha+MCup7Fb6otrfKIzyId2LW4rtnfeaykIVlri0uAKwoSqu2g0kUBb8QcP657OMrxjuPYf7m59k'
        '8xWo8rMvrgHdac7Cn1PpkGakTZepGVdKMTdZ2CqrNmM9j7KEc/NNKvABFMgwmcJfgc/ESUPQ77X4fKCSYi88JA0mWdFUW8E163K2'
        'AOSXLb6AZJsAnK7hVaVoivBVap20dGgxd6FfvW5cm1HcNnxZ12Rzp2ub3GMBE1aOrrgnD8Jk1Z58g4prKWP8OBq6brWbpy39mLuq'
        'Pait9d0y89HoQicflQ+bwgTqXzbVTTpZtEaSag5BbWfgrmNC/Ipuo8P6lBwgG1u8+Xq5yuxiQ+BqYMHSyWunYSfUKjyqbzmerRWp'
        'x+BUfiHagF4WK3blwCsLdD6pWjvyNxUYV++96NonL65aAyoAyU5uwZo8pE+wtkOq7BAA5LTP458xCGqBh+cjPjw/Oj0ReER/PL2Y'
        'ovxv7bawfqbDei73XX7mlH5uO7ocylmIH4JX5tkYj9IKbd3yrfNW02TcWauHvgUq3pZfy/qOSuxVvsGK9cfqmVkvtjtMkVhxrj4d'
        'W7PFnC2ArVaFF7os3V6LA79vfoT67/LpOgnS+RFqjHS9yhtfDf1nXDzD6sczsL+lutY793b4/vSrL/f/f3z1+r8BYC5ZrA==',
    'yolo_mask_auto_select_modules/config.py':
        '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'
        '/of/ArGidPY=',
    'yolo_mask_auto_select_modules/cli.py':
        'eNqdV21v4jgQ/s6vGOUTlSCF7ouqVpzEttxudbQgoLe6+xKZZEh8Teyc7ZSyv/7GTgJ9Sejt8mG32J6ZZ2aeecHzvCuZZUxE/ZQL'
        'BKbiIkNhIGdKcxH7nud1NkpmEASbwhQKgwB4lktlgAkhDTNcCt3p1GcqtpJYyuTMJClf1wJz+topb/xQig2P65tuB+hzPfl9fD9d'
        'BeO7q2+zxTK4Wv7Ze3FxfbNc0eVk+fL45nb8dbKk20V5Ph3fjoP5eDGeTifTYL6YkcSyFprPlqvqKPg+W/wxWbzSNrtfze9XB231'
        'OQmtJleryXXw9+xu0oDu++Tm67cVaTvpdCLcuBBiQBHR3VzJfzA0Fy4EJ9D/bR8o/45lqHMW4oVTlsPocDeusjG331Q3Qh0qntuI'
        'j7xxYWRG0Q9Zmu4glBmJIPw1m84gY/oBIq4NEyFqYDHjQhv4Ip/6UtDjwmYWpuyW+d5JadVnURTUye96/T7PWIy6H3Hl9cDschxZ'
        '6D0gx1iRmlFb7F99EkzzkTdTPOaCpaBloUIEpxw2Mo1QtSIg8IlU+rj5N1RptG9lwUioVAK9hg1PsdU25ctQvjDq/5AC38HQSoxG'
        'LNb2lhOevQ1wNsAkzICQEFIx8ogZpDxSYhMmYrwEr0mj+3gVnouDxsBp9EP9CAKfjHXdJPjc/R7wDXAD+EQ00a1xELIhFCwsGaiN'
        'pHZgVIHeEX9vYkHP3niLjygsBuaSwTWxoRBRK5At8jgx7yRiX4DvEsGVicbYqncdDGoDR+uBzFd6lKTK0uDKySQE3933AP3Yh2ww'
        '+BgMhp/8XMSXsE8PFWqVgvJ1u7GUZ9zUvnJhDq4OagQz1waopEKWEwxyJi3DW6qG7sA2kjQ9aTWiiGcy62uW5Sm2GGuP5NLZK32n'
        '8bGDUl1l/hIU/ltwRUCI0VX0gByvfWsDlXHRz6rm04jpwxFMt1zwrMhAFNkaFcgNlM7tg7JNpEZX+YK6LmiDGf3DlNFlSd6eZtDd'
        '5+tDe+w0YtQI72xw9nk4GJ4fjRwB2lAsFObIDFunWAfPweWONGrXV4UgvkT0jtJsv+g65mhf4nss2k8A722pvJqkzZ3KLgbkKg0W'
        'ZpmV8ydMD3PlGbMHvbPex97n3nlvODhSQbH+0Ri04dn5oN2J1BVoc7TbpNx2ESHBrOU2qWTPqe0PPh1xfSELY2ekSYjCCc0pH1aW'
        '6eX8hbBQiixR7esitwuMBqvQtYP2kRrhI3eAqAXQH3rU9RgNcTrwwiJi7v+88E4OKKv7dpyuj3GxQWV9hcpCa1thlM8GFG3W3Xlr'
        'pcoI02fM8qz2fYO8mc1p6TDgXoGttuOo8rKhthT90YWuhbuiShLUUPYmIKfmsGYmTKA7pC6pqVXRQ87SHlCqYY0Je+S0qbTXP9Ej'
        'L8zxafR6j2zEeU09MqQ5unMdYcuFsLSzC5zu2QbhuGWbQITrInatq73gc6lN5WV/K9UDquZwtq7AzQO87J12w5SaUyM41SFN9IhO'
        'xKEQfjWStGE41w49/+fWC5rEUeEkHNcuQD/wHMIUmbAQ97vU/O4rRZQgG9IPOkGsAisfUaVsp49tWN29nsD5Xq5WVtpmjNjkEgZ2'
        'AdeG05jfKm4MiqNuUy7TPqWYuRX757y+5trNDasjKHX4eckhN5HttOiOT79Uvwu4luKdFERF/iuxd1uQ7bTuh8iL3dVGpB4eliD/'
        'kxDV5GsGUwoppB+igmSf/cI6+Q+KfpkN',
    'yolo_mask_auto_select_modules/tests/test_dedup_and_fill_repair.py':
        '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',
}
MANIFEST_DATA = {
  "expected_original_sha256": {
    "yolo_mask_auto_select_modules/pipeline.py": "1b3ce85d4877a47e86c8fce7417562f2fd80046abb1fb8208c1b10cea9cfe466",
    "yolo_mask_auto_select_modules/config.py": "418df399b7b947cf842ca920f1f5f34fd0fda17b5fec47cd6019f204c18fb80c",
    "yolo_mask_auto_select_modules/cli.py": "9ab807634078b983dfe9c61ecf2482a90d8a81ce89e465e798ce474db19eb663",
    "yolo_mask_auto_select_modules/tests/test_dedup_and_fill_repair.py": "a3d8181485f34b941961ee5305b2dbed2af241038f67531253f66bcd3575578f"
  },
  "new_files": []
}


def payload_bytes(rel: str) -> bytes:
    import base64, zlib
    return zlib.decompress(base64.b64decode(EMBEDDED_PAYLOAD[rel]))

EXCLUDE_DIRS = {".venv", ".git", "__pycache__", "test_runs", "diagnostics", "yolo_mask_winners",
                "node_modules", ".pytest_cache"}


def fail(msg: str) -> None:
    print(f"\nABBRUCH: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if not (set(p.relative_to(root).parts) & EXCLUDE_DIRS):
            yield p


# ------------------------------------------------------------------ phase 1
def inventory(project: Path) -> dict:
    defs: dict[str, list[str]] = {}
    refs: set[str] = set()
    files = {}
    for p in py_files(project):
        rel = str(p.relative_to(project))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            fail(f"Syntaxfehler in {rel}: {exc}")
        imports = sorted({
            (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [None])
        })
        top = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        for n in top:
            defs.setdefault(n, []).append(rel)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                refs.add(n.id)
            elif isinstance(n, ast.Attribute):
                refs.add(n.attr)
            elif isinstance(n, ast.ImportFrom):
                refs.update(a.name for a in n.names)
        abs_paths = sorted({
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and (n.value.startswith(("/", "~/")) and len(n.value) > 3)
        })
        files[rel] = {"lines": len(p.read_text(encoding="utf-8").splitlines()),
                      "imports": imports, "top_level_defs": top, "absolute_path_literals": abs_paths}
    unreferenced = {n: where for n, where in defs.items()
                    if n not in refs and not n.startswith(("test_", "Test", "_test", "main"))}
    return {"files": files, "possibly_unreferenced_defs (heuristic, NOT a removal proof)": unreferenced,
            "test_files": [f for f in files if "/test" in f or f.startswith("test")],
            "config_files": [str(p.relative_to(project)) for p in project.glob("*.toml")]
                            + [str(p.relative_to(project)) for p in project.glob("requirements*.txt")]}


# ------------------------------------------------------------------ phase 2
def make_backup(project: Path, root: Path) -> Path:
    dest = root / f"{project.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    if dest.exists():
        fail(f"Backup existiert bereits, wird nie ueberschrieben: {dest}")
    try:
        dest.resolve().relative_to(project.resolve())
        fail("Backup-Verzeichnis darf nicht innerhalb des Projekts liegen.")
    except ValueError:
        pass
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project, dest, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS, "*.pyc"))
    return dest


# ------------------------------------------------------------------ phase 3
def check_patch_preconditions(project: Path) -> tuple[dict, list[str]]:
    m = MANIFEST_DATA
    problems = []
    for rel, expected in m["expected_original_sha256"].items():
        target = project / rel
        if not target.is_file():
            problems.append(f"fehlt: {rel}")
        elif sha256(target) != expected:
            if sha256(target) == hashlib.sha256(payload_bytes(rel)).hexdigest():
                continue  # already patched
            problems.append(f"Datei weicht vom analysierten Stand ab (nicht automatisch patchbar): {rel}")
    for rel in m["new_files"]:
        t = project / rel
        if t.exists() and sha256(t) != hashlib.sha256(payload_bytes(rel)).hexdigest():
            problems.append(f"Neue Datei wuerde bestehende, andere Datei ueberschreiben: {rel}")
    return m, problems


def apply_patch(project: Path, manifest: dict) -> list[str]:
    changed = []
    for rel in [*manifest["expected_original_sha256"], *manifest["new_files"]]:
        data, dst = payload_bytes(rel), project / rel
        if dst.is_file() and dst.read_bytes() == data:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        changed.append(rel)
    return changed


# ------------------------------------------------------------------ phase 5
def validate(project: Path, python: str) -> dict:
    res = {}
    comp = subprocess.run([python, "-m", "compileall", "-q", "yolo_mask_auto_select_modules",
                           "yolo_mask_auto_select.py", "yolo_mask_routing_audit.py"],
                          cwd=project, capture_output=True, text=True)
    res["syntax"] = {"ok": comp.returncode == 0, "output": (comp.stdout + comp.stderr)[-800:]}
    tests = subprocess.run([python, "-m", "unittest", "discover", "-s",
                            "yolo_mask_auto_select_modules/tests", "-t", ".", "-v"],
                           cwd=project, capture_output=True, text=True)
    tail = (tests.stdout + tests.stderr).strip().splitlines()[-6:]
    res["unit_tests"] = {"ok": tests.returncode == 0, "tail": tail}
    return res


def restore(backup: Path, project: Path) -> None:
    m = MANIFEST_DATA
    for rel in m["expected_original_sha256"]:
        shutil.copy2(backup / rel, project / rel)
        print(f"wiederhergestellt: {rel}")
    for rel in m["new_files"]:
        t = project / rel
        if t.exists() and not (backup / rel).exists():
            t.rename(t.with_suffix(t.suffix + f".removed_{time.strftime('%Y%m%d_%H%M%S')}"))
            print(f"zur Seite gelegt: {rel}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, default=next((c for c in (HERE, HERE.parent) if (c / "yolo_mask_auto_select_modules").is_dir()),
                                 Path.home() / "Downloads" / "Cleanup"))
    ap.add_argument("--backup-root", type=Path, default=Path.home() / "cleanup_backups")
    ap.add_argument("--apply", action="store_true", help="Backup, patch and validate (default: dry run)")
    ap.add_argument("--restore", type=Path, metavar="BACKUP_DIR")
    ap.add_argument("--python", default=sys.executable, help="Interpreter of the project .venv for the tests")
    args = ap.parse_args()
    project = args.project.expanduser().resolve()
    if not (project / "yolo_mask_auto_select_modules" / "pipeline.py").is_file():
        fail(f"Kein Cleanup-Projekt unter {project} (--project angeben).")

    if args.restore:
        restore(args.restore.expanduser().resolve(), project)
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = project / "optimization_reports" / stamp
    report_dir.mkdir(parents=True, exist_ok=False)

    print("[Phase 1] Bestandsaufnahme ...")
    inv = inventory(project)
    (report_dir / "inventory.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {len(inv['files'])} Python-Dateien, {len(inv['test_files'])} Testdateien")

    print("[Phase 3] Voraussetzungen des Patches pruefen ...")
    manifest, problems = check_patch_preconditions(project)
    if problems:
        fail("Patch wird nicht angewendet:\n  - " + "\n  - ".join(problems))
    print("  alle Zieldateien entsprechen dem analysierten Stand.")

    if not args.apply:
        print(f"\nDRY RUN beendet. Bericht: {report_dir}\nMit --apply ausfuehren, um Backup + Patch durchzufuehren.")
        return 0

    print("[Phase 2] Backup ...")
    backup = make_backup(project, args.backup_root.expanduser().resolve())
    print(f"  {backup}")

    print("[Phase 3] Patch anwenden ...")
    changed = apply_patch(project, manifest)
    for rel in changed:
        print(f"  geaendert: {rel}")

    print("[Phase 5] Validierung ...")
    val = validate(project, args.python)
    for k, v in val.items():
        print(f"  {k}: {'OK' if v['ok'] else 'FEHLER'}")
    ok = all(v["ok"] for v in val.values())

    lines = [f"# Optimierungsbericht {stamp}", "", f"Backup: `{backup}`", "",
             "## Geaenderte Dateien", *[f"- `{c}`" for c in changed], "",
             "## Validierung", *[f"- {k}: {'OK' if v['ok'] else 'FEHLER'}" for k, v in val.items()], "",
             "```", *val["unit_tests"]["tail"], "```", "",
             "## Nicht Teil dieser Stufe",
             "- Umstrukturierung nach src/cleanup/, pyproject.toml/uv.lock (erst nach Baseline-Messung)",
             "- Laufzeit-/VRAM-Vergleich vorher/nachher: erfordert echten Lauf mit GPU "
             "(`run_stats.json` im Ausgabeordner liefert die Phasenzeiten)",
             "- Bewertung der Cleaning-Methoden/Scoring-Metriken: erfordert die vollstaendigen candidate_scores.csv",
             "",
             "Wiederherstellung: `python optimize_cleanup.py --restore " + str(backup) + "`"]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nBericht: {report_dir / 'report.md'}")
    if not ok:
        print(f"FEHLER bei der Validierung. Wiederherstellen mit:\n  python {Path(__file__).name} --restore {backup}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
