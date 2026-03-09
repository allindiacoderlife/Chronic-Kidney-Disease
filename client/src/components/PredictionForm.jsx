import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Loader2 } from 'lucide-react'

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    age: '',
    bp: '',
    sg: '',
    al: '',
    su: '',
    rbc: '',
    pc: '',
    pcc: '',
    ba: '',
    bgr: '',
    bu: '',
    sc: '',
    sod: '',
    pot: '',
    hemo: '',
    pcv: '',
    wc: '',
    rc: '',
    htn: '',
    dm: '',
    cad: '',
    appet: '',
    pe: '',
    ane: ''
  })

  const handleInputChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    
    // Convert string values to numbers where needed
    const processedData = {
      age: parseFloat(formData.age) || 0,
      bp: parseFloat(formData.bp) || 0,
      sg: parseFloat(formData.sg) || 0,
      al: parseFloat(formData.al) || 0,
      su: parseFloat(formData.su) || 0,
      rbc: formData.rbc,
      pc: formData.pc,
      pcc: formData.pcc,
      ba: formData.ba,
      bgr: parseFloat(formData.bgr) || 0,
      bu: parseFloat(formData.bu) || 0,
      sc: parseFloat(formData.sc) || 0,
      sod: parseFloat(formData.sod) || 0,
      pot: parseFloat(formData.pot) || 0,
      hemo: parseFloat(formData.hemo) || 0,
      pcv: formData.pcv,
      wc: formData.wc,
      rc: formData.rc,
      htn: formData.htn,
      dm: formData.dm,
      cad: formData.cad,
      appet: formData.appet,
      pe: formData.pe,
      ane: formData.ane
    }
    
    onSubmit(processedData)
  }

  const fillSampleData = () => {
    setFormData({
      age: '48',
      bp: '80',
      sg: '1.02',
      al: '1',
      su: '0',
      rbc: 'normal',
      pc: 'normal',
      pcc: 'notpresent',
      ba: 'notpresent',
      bgr: '121',
      bu: '36',
      sc: '1.2',
      sod: '138',
      pot: '4.4',
      hemo: '15.4',
      pcv: '44',
      wc: '7800',
      rc: '5.2',
      htn: 'yes',
      dm: 'yes',
      cad: 'no',
      appet: 'good',
      pe: 'no',
      ane: 'no'
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Patient Information</h2>
        <Button type="button" variant="outline" onClick={fillSampleData}>
          Fill Sample Data
        </Button>
      </div>

      {/* Demographic Information */}
      <Card>
        <CardHeader>
          <CardTitle>Demographics</CardTitle>
          <CardDescription>Basic patient information</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="age">Age (years)</Label>
            <Input
              id="age"
              type="number"
              step="0.1"
              value={formData.age}
              onChange={(e) => handleInputChange('age', e.target.value)}
              placeholder="e.g., 48"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="bp">Blood Pressure (mm/Hg)</Label>
            <Input
              id="bp"
              type="number"
              step="0.1"
              value={formData.bp}
              onChange={(e) => handleInputChange('bp', e.target.value)}
              placeholder="e.g., 80"
              required
            />
          </div>
        </CardContent>
      </Card>

      {/* Urine Test Results */}
      <Card>
        <CardHeader>
          <CardTitle>Urine Test Results</CardTitle>
          <CardDescription>Laboratory test results from urine analysis</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="sg">Specific Gravity</Label>
            <Input
              id="sg"
              type="number"
              step="0.001"
              value={formData.sg}
              onChange={(e) => handleInputChange('sg', e.target.value)}
              placeholder="e.g., 1.02"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="al">Albumin</Label>
            <Input
              id="al"
              type="number"
              step="0.1"
              value={formData.al}
              onChange={(e) => handleInputChange('al', e.target.value)}
              placeholder="e.g., 1"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="su">Sugar</Label>
            <Input
              id="su"
              type="number"
              step="0.1"
              value={formData.su}
              onChange={(e) => handleInputChange('su', e.target.value)}
              placeholder="e.g., 0"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="rbc">Red Blood Cells</Label>
            <Select value={formData.rbc} onValueChange={(value) => handleInputChange('rbc', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="normal">Normal</SelectItem>
                <SelectItem value="abnormal">Abnormal</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="pc">Pus Cell</Label>
            <Select value={formData.pc} onValueChange={(value) => handleInputChange('pc', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="normal">Normal</SelectItem>
                <SelectItem value="abnormal">Abnormal</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="pcc">Pus Cell Clumps</Label>
            <Select value={formData.pcc} onValueChange={(value) => handleInputChange('pcc', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="present">Present</SelectItem>
                <SelectItem value="notpresent">Not Present</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="ba">Bacteria</Label>
            <Select value={formData.ba} onValueChange={(value) => handleInputChange('ba', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="present">Present</SelectItem>
                <SelectItem value="notpresent">Not Present</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Blood Test Results */}
      <Card>
        <CardHeader>
          <CardTitle>Blood Test Results</CardTitle>
          <CardDescription>Laboratory test results from blood analysis</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="bgr">Blood Glucose Random (mg/dL)</Label>
            <Input
              id="bgr"
              type="number"
              step="0.1"
              value={formData.bgr}
              onChange={(e) => handleInputChange('bgr', e.target.value)}
              placeholder="e.g., 121"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="bu">Blood Urea (mg/dL)</Label>
            <Input
              id="bu"
              type="number"
              step="0.1"
              value={formData.bu}
              onChange={(e) => handleInputChange('bu', e.target.value)}
              placeholder="e.g., 36"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="sc">Serum Creatinine (mg/dL)</Label>
            <Input
              id="sc"
              type="number"
              step="0.1"
              value={formData.sc}
              onChange={(e) => handleInputChange('sc', e.target.value)}
              placeholder="e.g., 1.2"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="sod">Sodium (mEq/L)</Label>
            <Input
              id="sod"
              type="number"
              step="0.1"
              value={formData.sod}
              onChange={(e) => handleInputChange('sod', e.target.value)}
              placeholder="e.g., 138"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="pot">Potassium (mEq/L)</Label>
            <Input
              id="pot"
              type="number"
              step="0.1"
              value={formData.pot}
              onChange={(e) => handleInputChange('pot', e.target.value)}
              placeholder="e.g., 4.4"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="hemo">Hemoglobin (g/dL)</Label>
            <Input
              id="hemo"
              type="number"
              step="0.1"
              value={formData.hemo}
              onChange={(e) => handleInputChange('hemo', e.target.value)}
              placeholder="e.g., 15.4"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="pcv">Packed Cell Volume (%)</Label>
            <Input
              id="pcv"
              type="text"
              value={formData.pcv}
              onChange={(e) => handleInputChange('pcv', e.target.value)}
              placeholder="e.g., 44"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="wc">White Blood Cell Count</Label>
            <Input
              id="wc"
              type="text"
              value={formData.wc}
              onChange={(e) => handleInputChange('wc', e.target.value)}
              placeholder="e.g., 7800"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="rc">Red Blood Cell Count</Label>
            <Input
              id="rc"
              type="text"
              value={formData.rc}
              onChange={(e) => handleInputChange('rc', e.target.value)}
              placeholder="e.g., 5.2"
              required
            />
          </div>
        </CardContent>
      </Card>

      {/* Medical History */}
      <Card>
        <CardHeader>
          <CardTitle>Medical History</CardTitle>
          <CardDescription>Patient's medical conditions and symptoms</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="htn">Hypertension</Label>
            <Select value={formData.htn} onValueChange={(value) => handleInputChange('htn', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="yes">Yes</SelectItem>
                <SelectItem value="no">No</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="dm">Diabetes Mellitus</Label>
            <Select value={formData.dm} onValueChange={(value) => handleInputChange('dm', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="yes">Yes</SelectItem>
                <SelectItem value="no">No</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="cad">Coronary Artery Disease</Label>
            <Select value={formData.cad} onValueChange={(value) => handleInputChange('cad', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="yes">Yes</SelectItem>
                <SelectItem value="no">No</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="appet">Appetite</Label>
            <Select value={formData.appet} onValueChange={(value) => handleInputChange('appet', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="good">Good</SelectItem>
                <SelectItem value="poor">Poor</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="pe">Pedal Edema</Label>
            <Select value={formData.pe} onValueChange={(value) => handleInputChange('pe', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="yes">Yes</SelectItem>
                <SelectItem value="no">No</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="ane">Anemia</Label>
            <Select value={formData.ane} onValueChange={(value) => handleInputChange('ane', value)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="yes">Yes</SelectItem>
                <SelectItem value="no">No</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-center">
        <Button type="submit" size="lg" disabled={isLoading} className="w-full md:w-auto min-w-[200px]">
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing...
            </>
          ) : (
            'Predict CKD Risk'
          )}
        </Button>
      </div>
    </form>
  )
}

export default PredictionForm
