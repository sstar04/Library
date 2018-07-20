Semplice interfaccia per la libreria TMEInspector

E' composta da due classi:
TGSimulator: Interaccia con i dati del simulatore
TMEInspector: che analizza i dati simulati passati da TGSimulator

TestUM.cpp e' una demo che utilizza la libreria TMEInspector. Viene creata un'istanza di TGSimulator usando un file vtk, essa contiene il set di dati simulato. Successivamente viene creata un'istanza di TMEInspector per estrarre informazioni da questi dati, ovvero avere informazioni dal microambiente. La prima informazione e' quella di trovare il bordo dello sferoide simulato.

