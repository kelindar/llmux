// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package wire

import (
	"testing"

	"github.com/buger/jsonparser"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateObject(t *testing.T) {
	cases := map[string][]byte{
		"empty":           nil,
		"null":            []byte("null"),
		"array":           []byte("[]"),
		"trailing":        []byte(`{"a":1} {}`),
		"duplicate":       []byte(`{"a":1,"a":2}`),
		"malformedNested": []byte(`{"a":{"b":[}`),
		"invalidUTF8":     []byte{'{', '"', 'a', '"', ':', '"', 0xff, '"', '}'},
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateObject(body)
			require.Error(t, err)
			if name == "null" || name == "array" {
				assert.ErrorIs(t, err, ErrNotObject)
			}
		})
	}

	require.NoError(t, ValidateObject([]byte(`{"a":[1,{"b":"c"}]}`)))
}

func TestDirectScalarValues(t *testing.T) {
	value, typ := directTestValue(t, `"line\n\u2603"`)
	text, err := String(value, typ)
	require.NoError(t, err)
	assert.Equal(t, "line\n☃", text)

	value, typ = directTestValue(t, `null`)
	text, err = String(value, typ)
	require.NoError(t, err)
	assert.Empty(t, text)

	value, typ = directTestValue(t, `true`)
	flag, err := Bool(value, typ)
	require.NoError(t, err)
	assert.True(t, flag)

	value, typ = directTestValue(t, `false`)
	flag, err = Bool(value, typ)
	require.NoError(t, err)
	assert.False(t, flag)

	value, typ = directTestValue(t, `null`)
	flag, err = Bool(value, typ)
	require.NoError(t, err)
	assert.False(t, flag)

	value, typ = directTestValue(t, `3`)
	count, err := Int(value, typ)
	require.NoError(t, err)
	assert.Equal(t, 3, count)

	value, typ = directTestValue(t, `1.0`)
	_, err = Int(value, typ)
	assert.Error(t, err)

	value, typ = directTestValue(t, `1.5`)
	rate, err := Float(value, typ)
	require.NoError(t, err)
	assert.Equal(t, 1.5, rate)

	value, typ = directTestValue(t, `null`)
	rate, err = Float(value, typ)
	require.NoError(t, err)
	assert.Zero(t, rate)

	value, typ = directTestValue(t, `["a",null,"☃"]`)
	values, err := Strings(value, typ)
	require.NoError(t, err)
	assert.Equal(t, []string{"a", "", "☃"}, values)

	values, err = Strings([]byte(`null`), jsonparser.Null)
	require.NoError(t, err)
	assert.Nil(t, values)
	_, err = Strings([]byte(`"nope"`), jsonparser.String)
	assert.ErrorIs(t, err, ErrType)
	_, err = Strings([]byte(`[1]`), jsonparser.Array)
	assert.ErrorIs(t, err, ErrType)
	_, err = Strings([]byte(`[}`), jsonparser.Array)
	assert.Error(t, err)

	value, typ = directTestValue(t, `"nope"`)
	_, err = Bool(value, typ)
	assert.Error(t, err)

	_, err = String([]byte(`"\x"`), jsonparser.String)
	assert.ErrorIs(t, err, ErrValue)
	_, err = Bool([]byte(`truth`), jsonparser.Boolean)
	assert.ErrorIs(t, err, ErrValue)
	_, err = Int([]byte(`1e3`), jsonparser.Number)
	assert.ErrorIs(t, err, ErrValue)
	_, err = Float([]byte(`1e`), jsonparser.Number)
	assert.ErrorIs(t, err, ErrValue)
	_, err = Int(value, typ)
	assert.ErrorIs(t, err, ErrType)
	_, err = Float(value, typ)
	assert.ErrorIs(t, err, ErrType)

	value, typ = directTestValue(t, `null`)
	count, err = Int(value, typ)
	require.NoError(t, err)
	assert.Zero(t, count)
}

func TestFileMediaValue(t *testing.T) {
	value, typ := directTestValue(t, `{"file_id":"file-1","filename":"a.txt"}`)
	media, filename, err := ParseFileMediaValue(Value{Raw: value, Type: typ}, "file")
	require.NoError(t, err)
	assert.Equal(t, "file-1", media.Ref)
	assert.Equal(t, "a.txt", filename)

	dataValue, dataType := directTestValue(t, `{"file_data":"YQ==","filename":"a.txt"}`)
	media, filename, err = ParseFileMediaValue(Value{Raw: dataValue, Type: dataType}, "file")
	require.NoError(t, err)
	assert.Equal(t, []byte{'a'}, media.Data)
	assert.Equal(t, "a.txt", filename)

	urlValue, urlType := directTestValue(t, `{"file_url":"https://example.com/a.txt"}`)
	media, filename, err = ParseFileMediaValue(Value{Raw: urlValue, Type: urlType}, "file")
	require.NoError(t, err)
	assert.Equal(t, "https://example.com/a.txt", media.URL)
	assert.Empty(t, filename)

	source := []byte(`{"type":"file","file_id":"file-1"}`)
	owned := Copy(source)
	source[2] = 'X'
	assert.Equal(t, `{"type":"file","file_id":"file-1"}`, string(owned))

	cases := map[string]Value{
		"wrongType":       {Raw: []byte(`[]`), Type: jsonparser.Array},
		"unknownField":    {Raw: []byte(`{"file_id":"file-1","extra":true}`), Type: jsonparser.Object},
		"malformed":       {Raw: []byte(`{"file_id":`), Type: jsonparser.Object},
		"missingSource":   {Raw: []byte(`{"filename":"a.txt"}`), Type: jsonparser.Object},
		"multipleSources": {Raw: []byte(`{"file_id":"file-1","file_url":"https://example.com/a"}`), Type: jsonparser.Object},
		"invalidFilename": {Raw: []byte(`{"file_id":"file-1","filename":1}`), Type: jsonparser.Object},
		"missingID":       {Raw: []byte(`{"file_id":""}`), Type: jsonparser.Object},
		"invalidData":     {Raw: []byte(`{"file_data":"!!!"}`), Type: jsonparser.Object},
		"invalidURL":      {Raw: []byte(`{"file_url":"ftp://example.com/a"}`), Type: jsonparser.Object},
	}
	for name, value := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, err := ParseFileMediaValue(value, "file")
			require.Error(t, err)
		})
	}
}

func directTestValue(t *testing.T, raw string) ([]byte, jsonparser.ValueType) {
	t.Helper()
	value, typ, _, err := jsonparser.Get([]byte(raw))
	require.NoError(t, err)
	return value, typ
}
