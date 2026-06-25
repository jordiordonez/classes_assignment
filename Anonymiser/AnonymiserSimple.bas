Attribute VB_Name = "AnonymiserSimple"
Option Explicit

Private Const SHEET_LISTE As String = "liste"
Private Const SHEET_DICT As String = "diccionari"

Public Sub Anonymiser()
    Dim wsListe As Worksheet
    Dim wsDict As Worksheet
    Dim lastRow As Long
    Dim lastDictRow As Long
    Dim rowNum As Long
    Dim colNum As Long
    Dim lastCol As Long
    Dim headerText As String
    Dim valueText As String
    Dim mappedValue As String
    Dim missing As String
    Dim answer As VbMsgBoxResult

    Set wsListe = ThisWorkbook.Worksheets(SHEET_LISTE)
    Set wsDict = EnsureDictionarySheet()

    answer = MsgBox( _
        "Cette action remplace directement les noms par les identifiants anonymes dans le fichier ouvert." & vbCrLf & vbCrLf & _
        "Gardez une copie originale avant de continuer." & vbCrLf & vbCrLf & _
        "Continuer ?", _
        vbQuestion + vbYesNo, _
        "Anonymiser")
    If answer <> vbYes Then Exit Sub

    FillDictionaryIfEmpty wsListe, wsDict

    lastRow = wsListe.Cells(wsListe.Rows.Count, "A").End(xlUp).Row
    lastDictRow = wsDict.Cells(wsDict.Rows.Count, "A").End(xlUp).Row
    lastCol = wsListe.Cells(1, wsListe.Columns.Count).End(xlToLeft).Column

    missing = ""
    For rowNum = 2 To lastRow
        For colNum = 1 To lastCol
            headerText = Trim(CStr(wsListe.Cells(1, colNum).Value))

            If IsTargetColumn(colNum, headerText) Then
                valueText = Trim(CStr(wsListe.Cells(rowNum, colNum).Value))
                If Len(valueText) > 0 Then
                    mappedValue = LookupInDictionary(wsDict, valueText, 1, 2, lastDictRow)
                    If Len(mappedValue) > 0 Then
                        wsListe.Cells(rowNum, colNum).Value = mappedValue
                    Else
                        wsListe.Cells(rowNum, colNum).Interior.Color = RGB(255, 199, 206)
                        missing = missing & "Ligne " & rowNum & ", colonne " & colNum & " : " & valueText & vbCrLf
                    End If
                End If
            End If
        Next colNum
    Next rowNum

    If Len(missing) > 0 Then
        MsgBox "Certains noms n'ont pas ete trouves dans le dictionnaire." & vbCrLf & vbCrLf & missing, vbExclamation, "Anonymiser"
    Else
        MsgBox "Anonymisation terminee dans le fichier ouvert." & vbCrLf & _
               "Ne pas envoyer le fichier si le full 'diccionari' est present.", vbInformation, "Anonymiser"
    End If
End Sub

Private Sub FillDictionaryIfEmpty(ByVal wsListe As Worksheet, ByVal wsDict As Worksheet)
    Dim lastRow As Long
    Dim rowNum As Long
    Dim dictRow As Long
    Dim nameText As String

    If Len(Trim(CStr(wsDict.Cells(2, 1).Value))) > 0 Then Exit Sub

    wsDict.Cells.Clear
    wsDict.Cells(1, 1).Value = "nom_real"
    wsDict.Cells(1, 2).Value = "id_anonim"
    wsDict.Columns(1).ColumnWidth = 40
    wsDict.Columns(2).ColumnWidth = 16

    lastRow = wsListe.Cells(wsListe.Rows.Count, "A").End(xlUp).Row
    dictRow = 2

    For rowNum = 2 To lastRow
        nameText = Trim(CStr(wsListe.Cells(rowNum, 1).Value))
        If Len(nameText) > 0 Then
            If Len(LookupInDictionary(wsDict, nameText, 1, 2, dictRow - 1)) = 0 Then
                wsDict.Cells(dictRow, 1).Value = nameText
                wsDict.Cells(dictRow, 2).Value = "eleve " & (dictRow - 1)
                dictRow = dictRow + 1
            End If
        End If
    Next rowNum
End Sub

Private Function EnsureDictionarySheet() As Worksheet
    On Error Resume Next
    Set EnsureDictionarySheet = ThisWorkbook.Worksheets(SHEET_DICT)
    On Error GoTo 0

    If EnsureDictionarySheet Is Nothing Then
        Set EnsureDictionarySheet = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
        EnsureDictionarySheet.Name = SHEET_DICT
        EnsureDictionarySheet.Cells(1, 1).Value = "nom_real"
        EnsureDictionarySheet.Cells(1, 2).Value = "id_anonim"
    End If
End Function

Private Function IsTargetColumn(ByVal colNum As Long, ByVal headerText As String) As Boolean
    If colNum = 1 Then
        IsTargetColumn = True
    Else
        IsTargetColumn = IsTargetHeader(headerText)
    End If
End Function

Private Function IsTargetHeader(ByVal headerText As String) As Boolean
    Select Case headerText
        Case "student", "Elèves à affecter", "Eleves a affecter", "avec1", "avec2", "sans1", "sans2", "sans3", "sans4", "sans5", "Source", "Other"
            IsTargetHeader = True
        Case Else
            IsTargetHeader = False
    End Select
End Function

Private Function LookupInDictionary( _
    ByVal wsDict As Worksheet, _
    ByVal searchText As String, _
    ByVal searchCol As Long, _
    ByVal returnCol As Long, _
    ByVal lastRow As Long) As String

    Dim rowNum As Long

    For rowNum = 2 To lastRow
        If Trim(CStr(wsDict.Cells(rowNum, searchCol).Value)) = searchText Then
            LookupInDictionary = Trim(CStr(wsDict.Cells(rowNum, returnCol).Value))
            Exit Function
        End If
    Next rowNum

    LookupInDictionary = ""
End Function
