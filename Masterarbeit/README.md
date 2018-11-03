Diese Datei enthält den Quellcode für die Reproduktion aller Abbildungen in der Masterarbeit

"Randomisierte lokalisierte Modellreduktion mit Robin-Transferoperator"
von Dennis Eickhorn

Um den Code auf einem Ubuntu-System ausführen zu können, geben Sie die folgenden Zeilen in eine Kommandozeile ein:


    sudo apt-get install python-pip python-virtualenv python-numpy python-scipy python-pyside cython python-matplotlib python-dev python git python-pil python-progress
    export PATH_TO_VIRTUALENV=~/pymor-virtualenv
    virtualenv --system-site-packages $PATH_TO_VIRTUALENV
    source $PATH_TO_VIRTUALENV/bin/activate
    export PYMOR_SOURCE_DIR=~/pymor
    git clone https://github.com/deneick/pymor $PYMOR_SOURCE_DIR
    cd $PYMOR_SOURCE_DIR
    git checkout Masterarbeit
    echo "$PYMOR_SOURCE_DIR/src" > $PATH_TO_VIRTUALENV/lib/python2.7/site-packages/pymor.pth
    python setup.py build_ext --inplace
    cd Masterarbeit


Um die Abbildungen zu reproduzieren, geben Sie den entsprechenden python-Befehl in die Kommandozentrale ein (Die Ergebnisse werden dann mit matplotlib visualisiert und sind im Ordner dats als .dat Dateien):

Abbildung 1:

    python 1_loesung_bsp1.py

Abbildung 2:

    python 2_loesung_bsp2.py

Abbildung 3:

    python 3_loesung_bsp3.py

Abbildung 5:

    python 5_FineGridResolution_a.py
    python 5_FineGridResolution_b.py
    python 5_FineGridResolution_c.py

Abbildung 6:

    python 17_cerr_a.py
    python 18_cerr_a.py

Abbildung 8:

    python 8_n_err_a.py
    python 8_n_err_b.py
    python 8_n_err_c.py

Abbildung 9:

    python 9_kerr_a.py
    python 9_kerr_b.py
    python 9_kerr_c.py

Abbildung 10:

    python 10_k_n_err_a.py
    python 10_k_n_err_b.py
    python 10_k_n_err_c.py

Abbildung 11:

    python 11_singularvalues.py

Abbildung 12:

    python 12_constants.py

Abbildung 13:

    python 13_kerr_adapt.py

Abbildung 14:

    python 14_tolglob_err.py

Abbildung 16:

    python 16_cerr_h_a.py
    python 16_cerr_h_b.py

Abbildung 17:

    python 17_cerr_a.py
    python 17_cerr_b.py
    python 17_cerr_c.py

Abbildung 18:

    python 18_cerr_a.py
    python 18_cerr_b.py
    python 18_cerr_c.py

Abbildung 19:

    python 19_cerr_a.py
    python 19_cerr_b.py
    python 19_cerr_c.py

Abbildung 20:
    
    python 20_cerr_a.py
    python 20_cerr_b.py
    python 20_cerr_c.py

Abbildung 21:

    python 21_cerr_a.py
    python 21_cerr_b.py
    python 21_cerr_c.py

Abbildung 22:

    python 22_kerr_per_a.py
    python 22_kerr_per_b.py
    python 22_kerr_per_c.py
