Attribute VB_Name = "DesanonymiserSimple"
Option Explicit

Private Const SHEET_DICT As String = "diccionari"

Public Sub Desanonymiser()
    Dim wsDict As Worksheet
    Dim ws As Worksheet
    Dim lastDictRow As Long
    Dim changed As Long
    Dim answer As VbMsgBoxResult

    Set wsDict = ThisWorkbook.Worksheets(SHEET_DICT)
    lastDictRow = wsDict.Cells(wsDict.Rows.Count, "B").End(xlUp).Row

    If lastDictRow < 2 Then
        MsgBox "Le full 'diccionari' est vide. Il faut les colonnes nom_real et id_anonim.", vbExclamation, "Desanonymiser"
        Exit Sub
    End If

    answer = MsgBox( _
        "Cette action remplace directement les identifiants anonymes par les noms reels dans le fichier ouvert." & vbCrLf & vbCrLf & _
        "Continuer ?", _
        vbQuestion + vbYesNo, _
        "Desanonymiser")
    If answer <> vbYes Then Exit Sub

    Application.ScreenUpdating = False
    For Each ws In ThisWorkbook.Worksheets
        If ws.Name <> SHEET_DICT Then
            changed = changed + DesanonymiseWorksheet(ws, wsDict, lastDictRow)
        End If
    Next ws
    Application.ScreenUpdating = True

    MsgBox "Desanonymisation terminee. Cellules modifiees : " & changed, vbInformation, "Desanonymiser"
End Sub

Public Sub ImporterEtDesanonymiser()
    Dim sourceWb As Workbook
    Dim sourceWs As Worksheet
    Dim destWs As Worksheet
    Dim wsDict As Worksheet
    Dim ws As Worksheet
    Dim lastDictRow As Long
    Dim changed As Long

    Set wsDict = ThisWorkbook.Worksheets(SHEET_DICT)
    lastDictRow = wsDict.Cells(wsDict.Rows.Count, "B").End(xlUp).Row

    If lastDictRow < 2 Then
        MsgBox "Le full 'diccionari' est vide. Copiez d'abord le dictionnaire nom_real / id_anonim.", vbExclamation, "Importer et desanonymiser"
        Exit Sub
    End If

    Set sourceWb = SelectOpenSourceWorkbook()
    If sourceWb Is Nothing Then
        MsgBox "Ouvrez d'abord le fichier assignments.xlsx genere par Streamlit, puis relancez cette macro.", vbExclamation, "Importer et desanonymiser"
        Exit Sub
    End If

    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    DeleteAllSheetsExceptDictionary

    For Each sourceWs In sourceWb.Worksheets
        sourceWs.Copy After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count)
        Set destWs = ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count)
        On Error Resume Next
        destWs.Name = sourceWs.Name
        On Error GoTo 0
    Next sourceWs

    For Each ws In ThisWorkbook.Worksheets
        If ws.Name <> SHEET_DICT Then
            changed = changed + DesanonymiseWorksheet(ws, wsDict, lastDictRow)
        End If
    Next ws

    Application.DisplayAlerts = True
    Application.ScreenUpdating = True

    MsgBox "Import et desanonymisation termines." & vbCrLf & _
           "Cellules modifiees : " & changed & vbCrLf & vbCrLf & _
           "Le fichier ouvert contient maintenant les onglets desanonymises.", _
           vbInformation, "Importer et desanonymiser"
End Sub

Private Function SelectOpenSourceWorkbook() As Workbook
    Dim wb As Workbook
    Dim candidates As Collection
    Dim message As String
    Dim answer As Variant
    Dim i As Long

    Set candidates = New Collection

    For Each wb In Application.Workbooks
        If wb.Name <> ThisWorkbook.Name Then
            candidates.Add wb
        End If
    Next wb

    If candidates.Count = 0 Then
        Set SelectOpenSourceWorkbook = Nothing
        Exit Function
    End If

    If candidates.Count = 1 Then
        Set SelectOpenSourceWorkbook = candidates.Item(1)
        Exit Function
    End If

    message = "Plusieurs fichiers Excel sont ouverts." & vbCrLf & _
              "Indiquez le numero du fichier assignments.xlsx a importer:" & vbCrLf & vbCrLf

    For i = 1 To candidates.Count
        message = message & i & " - " & candidates.Item(i).Name & vbCrLf
    Next i

    answer = InputBox(message, "Choisir le fichier source")

    If Len(Trim(CStr(answer))) = 0 Then
        Set SelectOpenSourceWorkbook = Nothing
        Exit Function
    End If

    If Not IsNumeric(answer) Then
        Set SelectOpenSourceWorkbook = Nothing
        Exit Function
    End If

    i = CLng(answer)
    If i < 1 Or i > candidates.Count Then
        Set SelectOpenSourceWorkbook = Nothing
        Exit Function
    End If

    Set SelectOpenSourceWorkbook = candidates.Item(i)
End Function

Private Sub DeleteAllSheetsExceptDictionary()
    Dim sheetIndex As Long

    If ThisWorkbook.Worksheets.Count = 1 Then Exit Sub

    For sheetIndex = ThisWorkbook.Worksheets.Count To 1 Step -1
        If ThisWorkbook.Worksheets(sheetIndex).Name <> SHEET_DICT Then
            ThisWorkbook.Worksheets(sheetIndex).Delete
        End If
    Next sheetIndex
End Sub

Private Function DesanonymiseWorksheet(ByVal ws As Worksheet, ByVal wsDict As Worksheet, ByVal lastDictRow As Long) As Long
    Dim lastCol As Long
    Dim lastRow As Long
    Dim colNum As Long
    Dim rowNum As Long
    Dim headerText As String
    Dim valueText As String
    Dim mappedValue As String

    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column

    For colNum = 1 To lastCol
        headerText = Trim(CStr(ws.Cells(1, colNum).Value))

        If IsTargetHeader(headerText) Then
            lastRow = ws.Cells(ws.Rows.Count, colNum).End(xlUp).Row

            For rowNum = 2 To lastRow
                valueText = Trim(CStr(ws.Cells(rowNum, colNum).Value))
                If Len(valueText) > 0 Then
                    mappedValue = LookupInDictionary(wsDict, valueText, 2, 1, lastDictRow)
                    If Len(mappedValue) > 0 Then
                        ws.Cells(rowNum, colNum).Value = mappedValue
                        DesanonymiseWorksheet = DesanonymiseWorksheet + 1
                    End If
                End If
            Next rowNum
        End If
    Next colNum

    If ws.Name = "Tableau" Then
        DesanonymiseWorksheet = DesanonymiseWorksheet + DesanonymiseTableau(ws, wsDict, lastDictRow)
    End If
End Function

Private Function DesanonymiseTableau(ByVal ws As Worksheet, ByVal wsDict As Worksheet, ByVal lastDictRow As Long) As Long
    Dim cell As Range
    Dim valueText As String
    Dim mappedValue As String

    For Each cell In ws.UsedRange.Cells
        valueText = Trim(CStr(cell.Value))
        If Len(valueText) > 0 Then
            mappedValue = LookupInDictionary(wsDict, valueText, 2, 1, lastDictRow)
            If Len(mappedValue) > 0 Then
                cell.Value = mappedValue
                DesanonymiseTableau = DesanonymiseTableau + 1
            End If
        End If
    Next cell
End Function

Private Function IsTargetHeader(ByVal headerText As String) As Boolean
    Select Case headerText
        Case "student", "Elèves à affecter", "Eleves a affecter", "avec1", "avec2", "sans1", "sans2", "Source", "Other"
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
